"""Causal Adaptive Multiscale Reliability Adapter (CARMA).

Attach after any forecaster returning [batch, horizon, channels]. All inputs
must already be in the same numerical scale; text has one embedding per
historical time step. No future tokens or target labels enter forward().
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CARMAConfig:
    text_dim: int
    channels: int = 1
    hidden: int = 32
    max_horizon: int = 336
    scales: tuple[int, ...] = (1, 4, 12)
    lags: tuple[int, ...] = (0, 1, 2)
    max_correction_sd: float = 0.5


class CARMA(nn.Module):
    """Output-level multimodal correction with causal scale/lag evidence.

    ``history`` and ``base_pred`` are [B,L,C] and [B,H,C]. ``text`` is
    [B,L,D]. ``text_mask`` [B,L] marks text available at each historical
    time step; absent text must be zero and masked. Horizon can vary up to
    config.max_horizon. Return has the same shape/scale as base_pred.
    """

    def __init__(self, config: CARMAConfig):
        super().__init__()
        if min(config.scales) < 1 or min(config.lags) < 0:
            raise ValueError("scales must be positive and lags nonnegative")
        self.config = config
        self.text_projection = nn.Linear(config.text_dim, config.channels, bias=False)
        self.evidence_projection = nn.Linear(config.channels * 3, config.hidden)
        self.query = nn.Parameter(torch.randn(config.hidden) * 0.02)
        self.horizon_embedding = nn.Embedding(config.max_horizon, config.hidden)
        self.correction_head = nn.Sequential(
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.channels),
        )
        # Starting from exact base output protects the existing trained model.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        self.logit_gate = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _causal_mean(x: Tensor, kernel: int) -> Tensor:
        if kernel == 1:
            return x
        left = F.pad(x.transpose(1, 2), (kernel - 1, 0), mode="replicate")
        return F.avg_pool1d(left, kernel_size=kernel, stride=1).transpose(1, 2)

    def forward(
        self,
        base_pred: Tensor,
        history: Tensor,
        text: Tensor,
        text_mask: Tensor,
    ) -> Tensor:
        if base_pred.ndim != 3 or history.ndim != 3 or text.ndim != 3:
            raise ValueError("base_pred, history, text must be rank-3 tensors")
        b, h, c = base_pred.shape
        if (history.shape[0], history.shape[2]) != (b, c):
            raise ValueError("history must match prediction batch and channels")
        if text.shape != (b, history.shape[1], self.config.text_dim):
            raise ValueError("text must be [B, L, text_dim] at history times")
        if text_mask.shape != (b, history.shape[1]):
            raise ValueError("text_mask must be [B, L]")
        if c != self.config.channels or h > self.config.max_horizon:
            raise ValueError("channels or horizon exceed CARMAConfig")
        if history.shape[1] < 2 or not torch.isfinite(history).all():
            raise ValueError("history needs at least 2 finite time steps")

        valid = text_mask.to(dtype=history.dtype).clamp(0, 1)
        safe_text = torch.where(valid[..., None] > 0, text, torch.zeros_like(text))
        if not torch.isfinite(safe_text).all():
            raise ValueError("available text embeddings must be finite")
        text_signal = self.text_projection(safe_text) * valid[..., None]
        center = history.mean(dim=1, keepdim=True).detach()
        spread = history.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        y = (history - center) / spread

        evidence = []
        reliabilities = []
        length = history.shape[1]
        for scale in self.config.scales:
            if scale > length:
                continue
            numeric = y - self._causal_mean(y, scale) if scale > 1 else y
            semantic = (
                text_signal - self._causal_mean(text_signal, scale)
                if scale > 1 else text_signal
            )
            for lag in self.config.lags:
                if lag >= length - 1:
                    continue
                # text[t-lag] can only explain numeric[t], never the reverse.
                n = numeric[:, lag:] if lag else numeric
                s = semantic[:, : length - lag] if lag else semantic
                m = valid[:, : length - lag] if lag else valid
                count = m.sum(dim=1, keepdim=True).clamp_min(1)
                n_mean = (n * m[..., None]).sum(1) / count
                s_mean = (s * m[..., None]).sum(1) / count
                n0 = (n - n_mean[:, None]) * m[..., None]
                s0 = (s - s_mean[:, None]) * m[..., None]
                covariance = (n0 * s0).sum(1) / count
                n_var = n0.square().sum(1) / count
                s_var = s0.square().sum(1) / count
                corr = covariance / (n_var * s_var).clamp_min(1e-8).sqrt()
                corr = corr.clamp(-1, 1)
                # Signed covariance distinguishes a useful negative relation.
                evidence.append(torch.cat((corr, n_mean, s_mean), dim=-1))
                reliabilities.append((m.sum(1) / (length - lag)).unsqueeze(-1))

        if not evidence:
            return base_pred
        tokens = torch.stack(evidence, dim=1)
        reliability = torch.stack(reliabilities, dim=1)
        bottleneck = torch.tanh(self.evidence_projection(tokens))
        logits = torch.einsum("bkd,d->bk", bottleneck, self.query)
        logits = logits + reliability.squeeze(-1).clamp_min(1e-6).log()
        weights = torch.softmax(logits, dim=1)
        context = (bottleneck * weights[..., None]).sum(dim=1)
        coverage = valid.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.logit_gate) * coverage
        step = self.horizon_embedding(torch.arange(h, device=history.device))
        correction = self.correction_head(context[:, None, :] + step[None, :, :])
        correction = torch.tanh(correction) * self.config.max_correction_sd
        return base_pred + gate[:, :, None] * correction * spread


class ForecastWithCARMA(nn.Module):
    """A common wrapper; the caller supplies the original model output."""

    def __init__(self, adapter: CARMA):
        super().__init__()
        self.adapter = adapter

    def forward(self, base_pred: Tensor, history: Tensor, text: Tensor, text_mask: Tensor) -> Tensor:
        return self.adapter(base_pred, history, text, text_mask)
