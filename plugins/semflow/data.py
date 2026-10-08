"""Shared, origin-aligned TimeMMD features for all five frozen forecasters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'benchmark/readgpt_data'
MODELS = ('TaTS', 'MM-TSFlib', 'SpecTF', 'CFA', 'Aurora')


def text_cache(domain: str, bert_path: Path, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Frozen BERT pooled token embeddings, one per timestamp; cache once/domain."""
    cache = Path(__file__).parent / 'cache' / f'{domain}_bert.npz'
    if cache.exists():
        with np.load(cache) as obj:
            return obj['embedding'], obj['mask']
    from transformers import AutoModel, AutoTokenizer
    frame = pd.read_csv(DATA / domain / f'{domain}.csv')
    raw = frame['fact'].fillna('').astype(str).tolist()
    usable = [bool(s.strip()) and s.strip().lower() != 'no information available' for s in raw]
    unique = list(dict.fromkeys(raw[i] for i, good in enumerate(usable) if good))
    tokenizer = AutoTokenizer.from_pretrained(str(bert_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(bert_path), local_files_only=True).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    vectors = {}
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=256,
                               return_tensors='pt').to(device)
            output = model(**tokens).last_hidden_state
            weights = tokens['attention_mask'].unsqueeze(-1)
            pooled = (output * weights).sum(1) / weights.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.update(zip(batch, pooled.cpu().numpy().astype(np.float32)))
            print(f'BERT {domain}: {min(start + batch_size, len(unique))}/{len(unique)}', flush=True)
    embedding = np.zeros((len(raw), model.config.hidden_size), dtype=np.float32)
    mask = np.asarray(usable, dtype=np.float32)
    for i, good in enumerate(usable):
        if good:
            embedding[i] = vectors[raw[i]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, embedding=embedding, mask=mask)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embedding, mask


class AdapterDataset(Dataset):
    def __init__(self, model: str, domain: str, horizon: int, split: str,
                 embedding: np.ndarray, text_mask: np.ndarray):
        path = ROOT / 'plugins/adapters' / model / domain / str(horizon) / f'{split}.npz'
        # 【阅读重点11：底模接口】V1-P1读取五模型预先导出的base_pred，不加载底模网络。
        with np.load(path) as obj:
            self.base = obj['base_pred'].astype(np.float32)
            self.target = obj['target'].astype(np.float32)
            self.history = obj['history'].astype(np.float32)
        manifest = json.loads((DATA / domain / 'manifest.json').read_text(encoding='utf-8'))
        self.seq_len = manifest['seq_len']
        start = manifest['train_end'] if split != 'test' else manifest['validation_end']
        if split == 'holdout':
            with np.load(path.with_name('fit.npz')) as fit:
                start += len(fit['target'])
        self.origins = np.arange(start, start + len(self.base), dtype=np.int64)
        self.embedding = embedding
        self.text_mask = text_mask
        if len(self.base) != len(self.target) or len(self.base) != len(self.history):
            raise ValueError(f'array length mismatch: {model}/{domain}/{horizon}/{split}')
        if len(self.origins) and self.origins[-1] > len(embedding):
            raise ValueError('text cache shorter than prediction origins')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        origin = self.origins[i]
        # 【阅读重点12：时间边界】origin是第一个待预测位置；条件只取[origin-L, origin)。
        # 若当前时点t=origin-1，则输入截至t，预测从t+1开始；包含当前文本而非未来文本。
        sl = slice(origin - self.seq_len, origin)
        return (self.base[i], self.target[i], self.history[i],
                self.embedding[sl], self.text_mask[sl])
