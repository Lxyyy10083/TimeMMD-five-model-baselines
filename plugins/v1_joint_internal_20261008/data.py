"""原始180任务的冻结数据；测试标签只在最终评估阶段读取。"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import pandas as pd
from ..internal_semflow_full_lab import data as aligned


def configure_inputs(inputs_root):
    root = Path(inputs_root).resolve()
    aligned.DATA = root / 'benchmark/readgpt_data'

    def frozen_text(domain, bert):
        cache = root / 'plugins/semflow/cache' / f'{domain}_bert.npz'
        if not cache.exists():
            raise FileNotFoundError(f'missing original frozen text cache: {cache}')
        with np.load(cache) as obj:
            return obj['embedding'].copy(), obj['mask'].copy()

    aligned.text_cache = frozen_text
    return root


class JointWindows(aligned.AlignedWindows):
    def __init__(self, inputs_root, domain, horizon, split, config, bert, aurora=False):
        root = configure_inputs(inputs_root)
        directory = aligned.DATA / domain
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        csv_path = directory / f'{domain}.csv'
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError(f'original data hash mismatch: {domain}')
        # pandas新版本使用ME/min别名；只调整数据时间特征解析，底模freq维度保持原配置。
        data_config = copy.copy(config)
        if data_config is not None:
            data_config.freq = {'m': 'ME', 'M': 'ME', 't': 'min'}.get(config.freq, config.freq)
        super().__init__(domain, horizon, split, data_config, bert, aurora, 'internal')
        # 【修改08】归一化只使用原始训练段，采用float64统计后转换成float32。
        values = pd.read_csv(csv_path).OT.to_numpy(dtype=np.float64)
        train = values[:manifest['train_end']]
        self.mean, self.std = float(train.mean()), float(train.std())
        if self.std < 1e-8:
            raise ValueError('constant target')
        self.numeric = ((values - self.mean) / self.std).astype(np.float32)[:, None]
        prior = pd.read_csv(csv_path).prior_history_avg.to_numpy(dtype=np.float64)
        self.prior = ((prior - self.mean) / self.std).astype(np.float32)
        self.inputs_root = root

    def original_test(self, model):
        # 【修改09】主对照是最初未加插件的保存预测，不替换为新重训control。
        path = self.inputs_root / 'plugins/adapters' / model / self.meta['domain'] / str(self.horizon) / 'test.npz'
        with np.load(path) as obj:
            return {k: obj[k].copy() for k in ('base_pred', 'target', 'history')}
