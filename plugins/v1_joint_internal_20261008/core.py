"""历史数值/文本 -> 内部融合 -> 条件未来分布 -> 可微点预测。

沿用V1的历史编码、滞后对齐和数值/语义混合思想。为了直接建模未来
数值密度，这里使用完整H维条件耦合流，不再截断为4个DCT系数。
H=1时就是下一时点条件分布。不是GANF原版异常检测模型的原样复现。
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..internal_semflow_full_lab.core import TrajectoryFlow, positions


class HistoryCondition(nn.Module):
    """【修改01】只读取预测起点之前的数值和文本，不读取训练target。"""
    def __init__(self, width=32, text_dim=768):
        super().__init__()
        self.numeric = nn.GRU(1, width, batch_first=True)
        self.text_projection = nn.Sequential(nn.Linear(text_dim, width), nn.LayerNorm(width), nn.Tanh())
        self.align = nn.Sequential(nn.Linear(3 * width + 1, width), nn.SiLU(), nn.Linear(width, 1))
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.query = nn.Linear(width, width)
        self.width = width

    def forward(self, history, text, mask):
        if history.ndim != 3 or history.shape[-1] != 1 or text.shape[:2] != history.shape[:2]:
            raise ValueError('expected aligned history[B,L,1], text[B,L,D]')
        if mask.shape != history.shape[:2]:
            raise ValueError('mask must be [B,L]')
        self.level = history.mean(1, keepdim=True).detach()
        self.scale = history.std(1, keepdim=True, unbiased=False).clamp_min(0.1).detach()
        self.numeric_states, _ = self.numeric((history - self.level) / self.scale)
        valid = mask.bool()
        projected = self.text_projection(torch.where(valid[..., None], text, torch.zeros_like(text)))
        choices, validity, logits = [], [], []
        novelty = (self.numeric_states - self.numeric_states[:, :1]).square().mean(-1, keepdim=True).add(1e-6).sqrt()
        # 【修改02】lag=0包含当前文本；lag>0只移动已经观测到的文本。
        for lag in (0, 1, 2):
            if lag >= history.shape[1]:
                continue
            shifted = F.pad(projected[:, :-lag], (0, 0, lag, 0)) if lag else projected
            active = F.pad(valid[:, :-lag], (lag, 0)) if lag else valid
            logits.append(self.align(torch.cat((self.numeric_states, shifted,
                self.numeric_states * shifted, novelty), -1)).squeeze(-1))
            choices.append(shifted)
            validity.append(active)
        lag_mask = torch.stack(validity, -1)
        weight = torch.softmax(torch.stack(logits, -1).masked_fill(~lag_mask, -1e4), -1) * lag_mask
        self.events = sum(weight[..., i, None] * value for i, value in enumerate(choices))
        self.valid = lag_mask.any(-1)
        self.coverage = valid.float().mean(-1, keepdim=True)
        time_weight = self.valid.float() / self.valid.sum(-1, keepdim=True).clamp_min(1)
        self.semantic = (self.events * time_weight[..., None]).sum(1)
        self.temporal = self.numeric_states[:, -1]
        return self.temporal

    def lookup(self, query):
        keys = self.key(self.events + positions(self.events.shape[1], self.width, query.device)[None])
        score = self.query(query) @ keys.transpose(1, 2) / math.sqrt(self.width)
        weight = score.masked_fill(~self.valid[:, None], -1e4).softmax(-1) * self.valid[:, None]
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        return weight @ self.value(self.events), weight

    def release(self):
        for name in ('level', 'scale', 'numeric_states', 'events', 'valid', 'coverage', 'semantic', 'temporal'):
            if hasattr(self, name):
                delattr(self, name)


class InternalAdapter(nn.Module):
    """【修改03】在原预测头之前修改隐藏表示，全部计算保留梯度。"""
    def __init__(self, feature_dim, width=32, complex_features=False):
        super().__init__()
        self.complex_features = complex_features
        incoming = 2 * feature_dim if complex_features else feature_dim
        self.down = nn.Linear(incoming, width)
        self.mix = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU())
        self.up = nn.Linear(width, incoming)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.strength = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features, memory):
        real = torch.cat((features.real, features.imag), -1) if self.complex_features else features
        shape = real.shape
        tokens = real.reshape(shape[0], -1, shape[-1])
        query = self.down(tokens)
        evidence, _ = memory.lookup(query)
        temporal = memory.temporal[:, None].expand_as(query)
        delta = self.up(self.mix(torch.cat((query, temporal, evidence), -1)))
        # 限制隐藏增量的相对尺度；这是内部适配，不是预测后的alpha回退。
        scale = tokens.square().mean(-1, keepdim=True).add(1e-8).sqrt().detach()
        output = tokens + 0.1 * self.strength.sigmoid() * scale * delta.tanh()
        output = output.reshape(shape)
        if self.complex_features:
            a, b = output.chunk(2, -1)
            return torch.complex(a, b)
        return output


class JointDistribution(nn.Module):
    """【修改04】完整未来残差密度及分布均值输出，NLL直接反传到主干。"""
    def __init__(self, horizon, feature_dim, width=32, samples=16):
        super().__init__()
        if horizon < 1 or samples < 2 or samples % 2:
            raise ValueError('horizon must be positive; samples must be positive and even')
        self.horizon = horizon
        self.memory = HistoryCondition(width)
        self.hidden_projection = nn.Linear(feature_dim, width)
        self.base_projection = nn.Linear(horizon, width)
        self.ts_context = nn.Sequential(nn.Linear(3 * width, width), nn.LayerNorm(width), nn.Tanh())
        self.joint_context = nn.Sequential(nn.Linear(4 * width, width), nn.LayerNorm(width), nn.Tanh())
        self.relevance = nn.Sequential(nn.Linear(4 * width + 1, width), nn.SiLU(), nn.Linear(width, 1))
        self.flow = TrajectoryFlow(horizon, width, layers=4, samples=samples)
        self.state = None

    def finish(self, native_prediction, hidden):
        if native_prediction.shape[1:] != (self.horizon, 1):
            raise ValueError('native prediction must be [B,H,1]')
        if torch.is_complex(hidden):
            hidden = torch.cat((hidden.real, hidden.imag), -1)
        # 【修改05】原生hidden和实时预测均不detach，密度条件参与主干学习。
        pooled = self.hidden_projection(hidden.reshape(hidden.shape[0], -1, hidden.shape[-1]).mean(1))
        base = self.base_projection(((native_prediction - self.memory.level) / self.memory.scale).squeeze(-1))
        temporal, semantic = self.memory.temporal, self.memory.semantic
        ts = self.ts_context(torch.cat((temporal, pooled, base), -1))
        joint = self.joint_context(torch.cat((temporal, semantic, pooled, base), -1))
        pos = positions(self.horizon, ts.shape[-1], ts.device, start=self.memory.numeric_states.shape[1])[None]
        ts, joint = ts[:, None] + pos, joint[:, None] + pos
        gate = self.relevance(torch.cat((temporal, semantic, pooled, base, self.memory.coverage), -1)).sigmoid()
        gate = gate * (self.memory.coverage > 0)
        ts_draws, joint_draws = self.flow.samples(torch.cat((ts, joint), 0)).chunk(2, 0)
        # 【修改06】使用概率混合的均值；点预测直接依赖flow参数，没有独立decision头。
        # 不对分布均值tanh限幅，确保点预测与实际预测分布一致。
        mean = (1 - gate) * ts_draws.mean(1) + gate * joint_draws.mean(1)
        prediction = native_prediction + self.memory.scale * mean[..., None]
        count = ts_draws.shape[1]
        samples = torch.cat((ts_draws, joint_draws), 1)
        weights = torch.cat(((1 - gate).expand(-1, count), gate.expand(-1, count)), 1) / count
        draws = native_prediction[:, None] + self.memory.scale[:, None] * samples[..., None]
        self.state = dict(base=native_prediction, prediction=prediction, ts_context=ts,
            joint_context=joint, gate=gate, draws=draws, sample_weights=weights)
        return prediction

    def density(self, target):
        st = self.state
        if st is None:
            raise RuntimeError('call forward before computing density')
        # 【修改07】监督真实未来残差；不detach base，NLL同时优化位置、条件和flow。
        residual = ((target - st['base']) / self.memory.scale).squeeze(-1)
        ts_nll = self.flow.nll(residual, st['ts_context'])
        joint_nll = self.flow.nll(residual, st['joint_context'])
        g = st['gate'].squeeze(-1).clamp(1e-6, 1 - 1e-6)
        mixed = -torch.logaddexp(torch.log1p(-g) - ts_nll, g.log() - joint_nll)
        # 无文本时精确采用数值条件密度，避免clamp造成微小语义混合。
        mixed = torch.where(self.memory.coverage.squeeze(-1) > 0, mixed, ts_nll)
        jacobian = self.horizon * self.memory.scale[:, 0, 0].log()
        return mixed + jacobian, ts_nll, joint_nll

    def objective(self, target, nll_weight=0.03, mae_weight=0.2,
                  utility_weight=0.01, regularity_weight=0.005):
        if target.shape != self.state['prediction'].shape:
            raise ValueError('target shape must match prediction')
        mixed_nll, ts_nll, joint_nll = self.density(target)
        error = self.state['prediction'] - target
        point = error.square().mean() + mae_weight * error.abs().mean()
        nll = mixed_nll.mean() / self.horizon
        teacher = (((ts_nll - joint_nll) / self.horizon - 0.05) / 0.25).sigmoid().detach()
        gate = self.state['gate'].squeeze(-1).clamp(1e-6, 1 - 1e-6)
        utility = (F.binary_cross_entropy(gate, teacher, reduction='none') *
                   (self.memory.coverage.squeeze(-1) > 0)).mean()
        regularity = (self.state['prediction'] - self.state['base']).square().mean()
        loss = point + nll_weight * nll + utility_weight * utility + regularity_weight * regularity
        return loss, {name: float(value.detach()) for name, value in
            dict(point=point, nll=nll, utility=utility, regularity=regularity).items()}

    def release(self):
        self.state = None
        self.memory.release()
