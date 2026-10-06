"""Semantic conditioned graph flow for future residual trajectories.

The base forecaster is untouched. Historical numeric/text evidence conditions
an invertible distribution over the next H residual trajectory coefficients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticGraphFlowConfig:
    text_dim: int = 768
    channels: int = 1
    hidden: int = 32
    coefficients: int = 4
    horizon: int = 12
    samples: int = 16
    lags: tuple[int, ...] = (0, 1, 2)
    correction_limit: float = 0.5
    correction_init: float = -2.0
    variance_beta: float = 1.0
    pooling: str = 'uniform'
    model_conditioned: bool = False
    utility_margin: float = 0.05


class TriangularGraphFlow(nn.Module):
    """Exact autoregressive density with ordered, learnable graph edges."""

    def __init__(self, size: int, hidden: int):
        super().__init__()
        self.size = size
        self.register_buffer('mask', torch.tril(torch.ones(size, size), diagonal=-1))
        self.edge_logits = nn.Parameter(torch.full((size, size), -2.0))
        self.node = nn.Embedding(size, hidden)
        self.net = nn.Sequential(nn.Linear(hidden + size, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def affine_parameters(self, values: Tensor, context: Tensor):
        edges = self.mask * torch.sigmoid(self.edge_logits)
        parents = values[:, None, :] * edges[None, :, :]
        node_context = context[:, None, :] + self.node.weight[None]
        params = self.net(torch.cat((node_context, parents), dim=-1))
        shift = params[..., 0]
        log_scale = 0.6 * torch.tanh(params[..., 1])
        return shift, log_scale

    def nll(self, values: Tensor, context: Tensor):
        shift, log_scale = self.affine_parameters(values, context)
        noise = (values - shift) * torch.exp(-log_scale)
        return (0.5 * (noise.square() + math.log(2 * math.pi)) + log_scale).sum(-1)

    def inverse(self, noise: Tensor, context: Tensor):
        values = torch.zeros_like(noise)
        for index in range(self.size):
            shift, log_scale = self.affine_parameters(values, context)
            next_value = noise[:, index] * log_scale[:, index].exp() + shift[:, index]
            values = values + next_value[:, None] * F.one_hot(
                torch.tensor(index, device=noise.device), self.size).to(noise.dtype)
        return values


class SemanticGraphFlow(nn.Module):
    """Common post-forecast module for MM-TSFlib, CFA, SpecTF, TaTS, Aurora.

    Inputs: base [B,H,1], history [B,L,1], text [B,L,D], mask [B,L].
    Output: point forecast, state with conditional distribution summaries.
    All numbers must use the same training-fitted normalization. Text must be
    published before the forecast origin; missing text is masked.
    """

    def __init__(self, config: SemanticGraphFlowConfig):
        super().__init__()
        if config.channels != 1:
            raise ValueError('v1 supports the univariate TimeMMD OT target')
        if config.coefficients > config.horizon or config.coefficients < 1:
            raise ValueError('coefficients must be between 1 and horizon')
        if not config.lags or min(config.lags) < 0:
            raise ValueError('lags must be nonempty and nonnegative')
        self.config = config
        width = config.hidden
        size = config.coefficients
        self.numeric = nn.GRU(1, width, batch_first=True)
        self.text_projection = nn.Sequential(nn.Linear(config.text_dim, width),
                                             nn.LayerNorm(width), nn.Tanh())
        self.align = nn.Sequential(nn.Linear(3 * width + 1, width), nn.SiLU(),
                                   nn.Linear(width, 1))
        self.ts_context = nn.Sequential(nn.Linear(width + size, width),
                                        nn.LayerNorm(width), nn.Tanh())
        self.joint_context = nn.Sequential(nn.Linear(2 * width + size, width),
                                           nn.LayerNorm(width), nn.Tanh())
        self.relevance = nn.Sequential(nn.Linear(2 * width + 2, width), nn.SiLU(),
                                       nn.Linear(width, 1))
        if config.pooling not in ('uniform', 'attention'):
            raise ValueError('pooling must be uniform or attention')
        if config.pooling == 'attention':
            self.time_attention = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(),
                                                 nn.Linear(width, 1))
        if config.model_conditioned:
            self.model_embedding = nn.Embedding(5, width)
            nn.init.zeros_(self.model_embedding.weight)
        self.flow = TriangularGraphFlow(size, width)
        self.correction_logit = nn.Parameter(torch.tensor(config.correction_init))
        time = torch.arange(config.horizon).float()[:, None] + 0.5
        freq = torch.arange(size).float()[None, :]
        basis = torch.cos(math.pi * time * freq / config.horizon) * math.sqrt(2 / config.horizon)
        basis[:, 0] /= math.sqrt(2)
        self.register_buffer('basis', basis)
        engine = torch.quasirandom.SobolEngine(size, scramble=True, seed=2026)
        uniform = engine.draw(max(1, config.samples // 2)).clamp(1e-5, 1 - 1e-5)
        z = math.sqrt(2) * torch.erfinv(2 * uniform - 1)
        self.register_buffer('noise', torch.cat((z, -z), dim=0))

    def coefficients(self, trajectory: Tensor):
        return trajectory.squeeze(-1) @ self.basis / math.sqrt(self.config.horizon)

    def sample(self, context: Tensor):
        b, width = context.shape
        count, size = self.noise.shape
        flat_context = context[:, None, :].expand(b, count, width).reshape(-1, width)
        flat_noise = self.noise[None, :, :].expand(b, count, size).reshape(-1, size)
        coeff = self.flow.inverse(flat_noise, flat_context).reshape(b, count, size)
        return math.sqrt(self.config.horizon) * (coeff @ self.basis.T)

    def _semantic_evidence(self, numeric_states: Tensor, text: Tensor, mask: Tensor):
        b, length, width = numeric_states.shape
        projected = self.text_projection(text)
        choices, validity = [], []
        for lag in self.config.lags:
            if lag >= length:
                continue
            shifted = F.pad(projected[:, :length - lag], (0, 0, lag, 0)) if lag else projected
            valid = F.pad(mask[:, :length - lag], (lag, 0)) if lag else mask
            novelty = ((numeric_states - numeric_states[:, :1]).square()
                       .mean(-1, keepdim=True) + 1e-6).sqrt()
            logits = self.align(torch.cat((numeric_states, shifted,
                                           numeric_states * shifted, novelty), -1)).squeeze(-1)
            choices.append((logits, shifted))
            validity.append(valid)
        logits = torch.stack([x[0] for x in choices], dim=-1)
        lag_mask = torch.stack(validity, dim=-1)
        logits = logits.masked_fill(lag_mask == 0, -1e4)
        lag_weights = torch.softmax(logits, dim=-1) * lag_mask
        aligned = sum(lag_weights[..., i, None] * choices[i][1]
                      for i in range(len(choices)))
        valid_times = lag_mask.any(-1).to(mask.dtype)
        time_weights = valid_times / valid_times.sum(dim=1, keepdim=True).clamp_min(1)
        if self.config.pooling == 'attention':
            query = numeric_states[:, -1:, :].expand_as(aligned)
            scores = self.time_attention(torch.cat((query, aligned, query * aligned), -1)).squeeze(-1)
            scores = scores.masked_fill(valid_times == 0, -1e4)
            time_weights = torch.softmax(scores, -1) * valid_times
            time_weights = time_weights / time_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        summary = (aligned * time_weights[..., None]).sum(dim=1)
        return summary, lag_weights

    def forward(self, base: Tensor, history: Tensor, text: Tensor, text_mask: Tensor,
                model_id: Tensor | None = None):
        b, h, c = base.shape
        if h != self.config.horizon or c != 1 or history.ndim != 3 or text.ndim != 3:
            raise ValueError('base/history/text shapes do not match configuration')
        if history.shape[:2] != text.shape[:2] or text.shape[-1] != self.config.text_dim:
            raise ValueError('text must align with historical numeric observations')
        if text_mask.shape != history.shape[:2]:
            raise ValueError('text_mask must be [B,L]')
        mask = text_mask.to(history.dtype).clamp(0, 1)
        safe_text = torch.where(mask[..., None] > 0, text, torch.zeros_like(text))
        level = history.mean(1, keepdim=True).detach()
        scale = history.std(1, keepdim=True, unbiased=False).clamp_min(0.1).detach()
        states, _ = self.numeric((history - level) / scale)
        temporal = states[:, -1]
        semantic, lag_weights = self._semantic_evidence(states, safe_text, mask)
        base_features = self.coefficients(((base - level) / scale).detach())
        ts_condition = self.ts_context(torch.cat((temporal, base_features), -1))
        joint_condition = self.joint_context(torch.cat((temporal, semantic, base_features), -1))
        if self.config.model_conditioned:
            if model_id is None:
                raise ValueError('model_id is required for a conditioned flow')
            model_state = self.model_embedding(model_id)
            ts_condition = ts_condition + model_state
            joint_condition = joint_condition + model_state
        coverage = mask.mean(1, keepdim=True)
        agreement = F.cosine_similarity(temporal, semantic, dim=-1).unsqueeze(-1)
        gate = torch.sigmoid(self.relevance(torch.cat((temporal, semantic,
                                                       coverage, agreement), -1))) * (coverage > 0)
        ts_draws, joint_draws = self.sample(torch.cat((ts_condition, joint_condition), 0)).chunk(2, 0)
        ts_mean, joint_mean = ts_draws.mean(1), joint_draws.mean(1)
        mixture_mean = (1 - gate) * ts_mean + gate * joint_mean
        ts_var = ts_draws.var(1, unbiased=False)
        joint_var = joint_draws.var(1, unbiased=False)
        mixture_var = ((1 - gate) * (ts_var + (ts_mean - mixture_mean).square())
                       + gate * (joint_var + (joint_mean - mixture_mean).square()))
        # Low confidence shrinks an approximate distribution's point correction.
        gain = torch.sigmoid(self.correction_logit) / (1 + self.config.variance_beta * mixture_var)
        correction = self.config.correction_limit * torch.tanh(mixture_mean) * gain
        pred = base + scale * correction[..., None]
        state = dict(ts_context=ts_condition, joint_context=joint_condition,
                     gate=gate.squeeze(-1), coverage=coverage.squeeze(-1),
                     lag_weights=lag_weights, mixture_mean=mixture_mean,
                     mixture_variance=mixture_var, ts_mean=ts_mean,
                     joint_mean=joint_mean, scale=scale)
        return pred, state

    def objective(self, pred: Tensor, target: Tensor, base: Tensor, state: dict,
                  nll_weight: float = 0.03, utility_weight: float = 0.01,
                  mae_weight: float = 0.2, regularity_weight: float = 0.005,
                  normalizers: tuple[Tensor, Tensor] | None = None):
        residual = self.coefficients((target - base.detach()) / state['scale'])
        ts_nll = self.flow.nll(residual, state['ts_context'])
        joint_nll = self.flow.nll(residual, state['joint_context'])
        gate = state['gate'].clamp(1e-6, 1 - 1e-6)
        log_mix = torch.logaddexp(torch.log1p(-gate) - ts_nll,
                                  torch.log(gate) - joint_nll)
        nll = -log_mix.mean() / self.config.coefficients
        # Teacher utility uses target only in the loss. Inference gate is causal.
        teacher = torch.sigmoid(((ts_nll - joint_nll) - self.config.utility_margin) / 0.25).detach()
        if not torch.isfinite(teacher).all() or not torch.isfinite(gate).all():
            raise FloatingPointError('nonfinite semantic gate or teacher')
        utility = F.binary_cross_entropy(gate, teacher, reduction='none')
        utility = (utility * (state['coverage'] > 0)).mean()
        error = pred - target
        if normalizers is None:
            point = error.square().mean() + mae_weight * error.abs().mean()
        else:
            mse_scale, mae_scale = normalizers
            point = ((error.square().mean((1, 2)) / mse_scale.clamp_min(1e-6)).mean()
                     + mae_weight * (error.abs().mean((1, 2)) / mae_scale.clamp_min(1e-6)).mean())
        regularity = (pred - base).square().mean()
        total = point + nll_weight * nll + utility_weight * utility + regularity_weight * regularity
        return total, dict(point=point.detach(), nll=nll.detach(), utility=utility.detach(),
                           regularity=regularity.detach())

    @staticmethod
    def semantic_profile(state: dict, history: Tensor, pred: Tensor):
        """Numerical semantic display; observational influence, not causality."""
        trend = pred.mean(1).squeeze(-1) - history[:, -1, 0]
        uncertainty = state['mixture_variance'].mean(1).sqrt()
        return dict(text_utility=state['gate'], expected_direction=torch.sign(trend),
                    expected_change=trend, model_uncertainty=uncertainty,
                    text_shift=(state['joint_mean'] - state['ts_mean']).mean(1))
