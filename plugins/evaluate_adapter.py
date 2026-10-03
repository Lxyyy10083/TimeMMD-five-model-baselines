"""Evaluate a frozen, validation-selected CARMA checkpoint on test once."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from carma_v1.carma import CARMA, CARMAConfig


def stats(pred, true):
    error = pred - true
    return {'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    data = np.load(args.directory / 'test.npz')
    saved = torch.load(args.directory / 'adapter.pt', map_location='cpu', weights_only=False)
    base = data['base_pred']
    if saved['enabled']:
        model = CARMA(CARMAConfig(**saved['config']))
        model.load_state_dict(saved['state_dict'])
        model.eval()
        with torch.no_grad():
            pred = model(*(torch.from_numpy(data[key]) for key in
                           ('base_pred', 'history', 'text', 'text_mask'))).numpy()
    else:
        pred = base
    report = {'enabled_by_validation': bool(saved['enabled']),
              'base_test': stats(base, data['target']),
              'adapter_test': stats(pred, data['target']),
              'base_val_mse_mae': saved['base_val_mse_mae'],
              'adapter_val_mse_mae': saved['adapter_val_mse_mae']}
    (args.directory / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.save(args.directory / 'pred_adapter.npy', pred)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
