"""Chronological train/validation/test windows shared by all native backbones."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from semflow.data import DATA, text_cache


class AlignedWindows(Dataset):
    def __init__(self, domain, horizon, split, config, bert, aurora=False, variant='internal'):
        directory = DATA / domain
        self.meta = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(directory / f'{domain}.csv')
        values = frame.OT.to_numpy(dtype=np.float32)
        boundary = self.meta['train_end']
        self.mean, self.std = float(values[:boundary].mean()), float(values[:boundary].std())
        if self.std < 1e-8:
            raise ValueError('constant training target')
        self.numeric = ((values - self.mean) / self.std).astype(np.float32)[:, None]
        self.embedding, self.mask = text_cache(domain, Path(bert))
        if len(values) != len(self.embedding):
            raise ValueError('canonical text cache length mismatch')
        self.horizon, self.length = horizon, self.meta['seq_len']
        self.label = config.label_len if config is not None else 0
        edges = {'train': (self.length, boundary),
                 'val': (boundary, self.meta['validation_end']),
                 'test': (self.meta['validation_end'], len(values))}
        start, end = edges[split]
        self.origins = np.arange(start, end - horizon + 1, dtype=np.int64)
        if not len(self.origins):
            raise ValueError(f'empty {domain}/{horizon}/{split}')
        self.prior = ((frame.prior_history_avg.to_numpy(dtype=np.float32) - self.mean) /
                      self.std).astype(np.float32)
        self.aurora, self.variant = aurora, variant
        if aurora:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(Path(bert)), local_files_only=True)
            self.tokens = []
            text = frame.fact.fillna('').astype(str).tolist()
            for o in self.origins:
                history = text[o-self.length:o]
                if variant == 'shuffle':
                    history = list(reversed(history))
                tokens = tokenizer(' '.join(history), padding='max_length',
                                   truncation=True, max_length=256, return_tensors='np')
                self.tokens.append({k: v[0].astype(np.int64) for k, v in tokens.items()})
        else:
            from utils.timefeatures import time_features
            freq = config.freq
            dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
            if config.embed == 'timeF':
                self.marks = time_features(dates, freq=freq).T.astype(np.float32)
            else:
                self.marks = np.stack((dates.month, dates.day, dates.dayofweek,
                                       dates.hour), -1).astype(np.float32)
                if freq in ('t', 'min'):
                    self.marks = np.column_stack((self.marks, dates.minute // 15)).astype(np.float32)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        o, h, l = int(self.origins[i]), self.horizon, self.length
        batch = dict(history=self.numeric[o-l:o], target=self.numeric[o:o+h],
                     text=self.embedding[o-l:o], mask=self.mask[o-l:o],
                     origin=np.int64(o), prior=np.full((h, 1), self.prior[o], np.float32))
        if self.aurora:
            batch.update(self.tokens[i])
            batch.setdefault('token_type_ids', np.zeros(256, dtype=np.int64))
        else:
            batch.update(marks=self.marks[o-l:o], future_marks=self.marks[o-self.label:o+h])
        return batch

