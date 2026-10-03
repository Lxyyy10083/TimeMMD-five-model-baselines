"""Train CARMA from original model predictions on train/validation only.

NPZ arrays: base_pred [N,H,C], history [N,L,C], text [N,L,D],
text_mask [N,L], target [N,H,C]. Values must be on the same scale.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from carma import CARMA, CARMAConfig


KEYS = ("base_pred", "history", "text", "text_mask", "target")


def read_npz(path):
    with np.load(path) as data:
        missing = set(KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        return {key: torch.from_numpy(data[key].astype(np.float32)) for key in KEYS}


def metrics(pred, target):
    err = pred - target
    return (err.square().mean().item(), err.abs().mean().item())


def evaluate(adapter, data, device):
    adapter.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in data.items()}
        pred = adapter(batch["base_pred"], batch["history"], batch["text"], batch["text_mask"])
        return metrics(pred, batch["target"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    train, val = read_npz(args.train), read_npz(args.val)
    for key in KEYS:
        if train[key].shape[1:] != val[key].shape[1:]:
            raise ValueError(f"train/val {key} shape mismatch")
    if len(train["target"]) == 0 or len(val["target"]) == 0:
        raise ValueError("train and val cannot be empty")
    config = CARMAConfig(
        text_dim=train["text"].shape[-1],
        channels=train["history"].shape[-1],
        max_horizon=train["target"].shape[1],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = CARMA(config).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    data = TensorDataset(*(train[key] for key in KEYS))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    base_metrics = metrics(val["base_pred"], val["target"])
    best_score = float("inf")
    best_state = None
    best_metrics = None
    stale = 0
    for epoch in range(args.epochs):
        adapter.train()
        for base, history, text, mask, target in loader:
            base, history, text, mask, target = (
                x.to(device) for x in (base, history, text, mask, target)
            )
            output = adapter(base, history, text, mask)
            error = output - target
            loss = error.square().mean() + 0.1 * error.abs().mean()
            # Penalize large corrections to reduce overfitting to text noise.
            loss = loss + 0.01 * (output - base).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        current = evaluate(adapter, val, device)
        score = current[0] / max(base_metrics[0], 1e-12) + current[1] / max(base_metrics[1], 1e-12)
        print(f"epoch={epoch + 1} val_mse={current[0]:.6g} val_mae={current[1]:.6g}", flush=True)
        if score < best_score:
            best_score, best_metrics, stale = score, current, 0
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    # Claim improvement only if both validation metrics beat identity.
    enabled = best_metrics[0] < base_metrics[0] and best_metrics[1] < base_metrics[1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "state_dict": best_state, "enabled": enabled,
                "base_val_mse_mae": base_metrics, "adapter_val_mse_mae": best_metrics}, args.out)
    print(f"enabled={enabled} base={base_metrics} adapter={best_metrics} out={args.out}")


if __name__ == "__main__":
    main()
