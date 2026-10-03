"""Historical event memory, internal feature conditioning and full-horizon flow."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def positions(length, width, device, start=0):
    t = torch.arange(start, start + length, device=device).float()[:, None]
    f = torch.exp(-math.log(10000) * torch.arange(0, width, 2, device=device).float() / width)
    p = torch.zeros(length, width, device=device)
    p[:, 0::2] = torch.sin(t * f)
    p[:, 1::2] = torch.cos(t * f[:p[:, 1::2].shape[1]])
    return p


class EventMemory(nn.Module):
    """Keep timestamps, numeric innovations and missingness in semantic queries."""
    def __init__(self, width=64, text_dim=768):
        super().__init__()
        self.width = width
        self.numeric = nn.GRU(2, width, batch_first=True)
        self.text = nn.Sequential(nn.Linear(text_dim, width), nn.LayerNorm(width), nn.SiLU())
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.relevance = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, history, text, mask):
        self.level = history.mean(1, keepdim=True).detach()
        self.scale = history.std(1, keepdim=True, unbiased=False).clamp_min(0.1).detach()
        x = (history - self.level) / self.scale
        diff = F.pad(x[:, 1:] - x[:, :-1], (0, 0, 1, 0))
        states, _ = self.numeric(torch.cat((x, diff), -1))
        valid = mask.bool()
        safe = torch.where(valid[..., None], text, torch.zeros_like(text))
        semantic = self.text(safe)
        gate = torch.sigmoid(self.relevance(torch.cat((states, semantic, states * semantic), -1)))
        self.numeric_states = states
        self.semantic = semantic
        self.event_states = semantic * gate
        self.valid = valid
        self.position = positions(history.shape[1], self.width, history.device)
        self.gate = gate
        return states

    def lookup(self, query):
        # A zero-text sample receives exactly zero semantic evidence, no NaN softmax.
        keys = self.key(self.event_states + self.position[None])
        scores = torch.einsum('bqd,bld->bql', self.query(query), keys) / math.sqrt(self.width)
        scores = scores.masked_fill(~self.valid[:, None], -1e4)
        weights = torch.softmax(scores, -1) * self.valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        values = self.value(self.event_states) * self.valid[..., None]
        return weights @ values, weights

    def horizon_context(self, horizon, hidden):
        numeric = self.numeric_states[:, -1:, :]
        q = numeric + hidden[:, None] + positions(horizon, self.width, hidden.device,
                                                 start=self.numeric_states.shape[1])[None]
        events, weights = self.lookup(q)
        return q + events, weights


class InternalFusion(nn.Module):
    """A hidden token queries historical events before the native forecast head."""
    def __init__(self, feature_dim, width=64, complex_features=False):
        super().__init__()
        self.complex_features = complex_features
        self.feature_dim = feature_dim
        incoming = 2 * feature_dim if complex_features else feature_dim
        self.down = nn.Linear(incoming, width)
        self.up = nn.Linear(width, incoming)
        self.modulate = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, memory):
        real = torch.cat((features.real, features.imag), -1) if self.complex_features else features
        shape = real.shape
        tokens = real.reshape(shape[0], -1, shape[-1])
        q = self.down(tokens)
        evidence, _ = memory.lookup(q)
        delta = self.up(self.modulate(torch.cat((q, evidence), -1)))
        active = memory.valid.any(-1).to(delta.dtype)[:, None, None]
        out = tokens + torch.sigmoid(self.strength) * delta * active
        out = out.reshape(shape)
        if self.complex_features:
            a, b = out.split(self.feature_dim, -1)
            return torch.complex(a, b)
        return out


class ConditionalCoupling(nn.Module):
    """Alternating horizon masks give an invertible full-dimensional density.

    Source values remain unchanged; only destination coordinates transform.
    Their conditioner sees source residuals and historical hidden context.
    """
    def __init__(self, horizon, width, parity):
        super().__init__()
        self.register_buffer('source', ((torch.arange(horizon) + parity) % 2 == 0).float()[None])
        self.context = nn.Linear(width, width)
        self.graph_key = nn.Linear(width, 16, bias=False)
        self.graph_query = nn.Linear(width, 16, bias=False)
        self.local = nn.Conv1d(1, width, 3, padding=1)
        self.graph = nn.Linear(1, width)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, 2))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def parameters_for(self, x, context):
        source_values = x * self.source
        scores = self.graph_query(context) @ self.graph_key(context).transpose(-1, -2) / 4
        scores = scores.masked_fill(self.source[:, None] == 0, -1e4)
        edges = scores.softmax(-1) * self.source[:, None]
        edges = edges / edges.sum(-1, keepdim=True).clamp_min(1e-8)
        global_source = edges @ source_values[..., None]
        hidden = self.context(context) + self.local(source_values[:, None]).transpose(1, 2)
        shift, log_scale = self.out(hidden + self.graph(global_source)).unbind(-1)
        destination = 1 - self.source
        return shift * destination, 0.8 * torch.tanh(log_scale) * destination

    def normalize(self, x, context):
        shift, log_scale = self.parameters_for(x, context)
        return (x - shift) * (-log_scale).exp(), -log_scale.sum(-1)

    def inverse(self, z, context):
        shift, log_scale = self.parameters_for(z, context)
        return z * log_scale.exp() + shift


class TrajectoryFlow(nn.Module):
    """No DCT truncation: model all H residual coordinates."""
    def __init__(self, horizon, width=64, layers=4, samples=16):
        super().__init__()
        self.layers = nn.ModuleList([ConditionalCoupling(horizon, width, i % 2)
                                     for i in range(layers)])
        engine = torch.quasirandom.SobolEngine(horizon, scramble=True, seed=2026)
        u = engine.draw(samples // 2).clamp(1e-5, 1 - 1e-5)
        z = math.sqrt(2) * torch.erfinv(2 * u - 1)
        self.register_buffer('noise', torch.cat((z, -z), 0))

    def nll(self, residual, context):
        z, jac = residual, residual.new_zeros(len(residual))
        for layer in self.layers:
            z, value = layer.normalize(z, context)
            jac = jac + value
        return (0.5 * (z.square() + math.log(2 * math.pi))).sum(-1) - jac

    def samples(self, context):
        b, h, w = context.shape
        count = len(self.noise)
        ctx = context[:, None].expand(b, count, h, w).reshape(-1, h, w)
        z = self.noise[None].expand(b, count, h).reshape(-1, h)
        for layer in reversed(self.layers):
            z = layer.inverse(z, ctx)
        return z.reshape(b, count, h)


class SemanticDistribution(nn.Module):
    def __init__(self, horizon, feature_dim, width=64):
        super().__init__()
        self.horizon = horizon
        self.memory = EventMemory(width)
        self.hidden_projection = nn.Linear(feature_dim, width)
        self.context = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU())
        self.flow = TrajectoryFlow(horizon, width)
        self.decision = nn.Linear(width, 1)
        self.state = None

    def finish(self, native_prediction, hidden):
        if torch.is_complex(hidden):
            hidden = torch.cat((hidden.real, hidden.imag), -1)
        tokens = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])
        # Never detach the native hidden state: NLL reaches the backbone.
        context, attention = self.memory.horizon_context(
            self.horizon, self.hidden_projection(tokens.mean(1)))
        context = self.context(context)
        draws = self.flow.samples(context)
        mean = draws.mean(1)
        median = torch.quantile(draws, 0.5, dim=1)
        decision = self.decision(context).sigmoid().squeeze(-1)
        correction = decision * mean + (1 - decision) * median
        scale = self.memory.scale
        pred = native_prediction + scale * correction[..., None]
        self.state = dict(base=native_prediction, context=context, draws=draws,
                          attention=attention, decision=decision, prediction=pred)
        return pred

    def objective(self, target, nll_weight=0.1, mae_weight=0.5,
                  energy_weight=0.05, regularity_weight=0.001):
        st = self.state
        error = st['prediction'] - target
        residual = ((target - st['base']) / self.memory.scale).squeeze(-1)
        # Residual location also has gradients into the native prediction.
        nll = (self.flow.nll(residual, st['context']) / self.horizon).mean()
        draws = st['draws']
        # Marginal energy/CRPS estimate with fixed paired draws (not independent MC).
        energy = (draws - residual[:, None]).abs().mean()
        energy = energy - 0.5 * (draws[:, :, None] - draws[:, None, :]).abs().mean()
        point = error.square().mean() + mae_weight * error.abs().mean()
        regularity = (st['prediction'] - st['base']).square().mean()
        loss = point + nll_weight * nll + energy_weight * energy + regularity_weight * regularity
        return loss, dict(point=float(point.detach()), nll=float(nll.detach()),
                          energy=float(energy.detach()), regularity=float(regularity.detach()))

    def release(self):
        self.state = None
        for name in ('level', 'scale', 'numeric_states', 'semantic', 'event_states',
                     'valid', 'position', 'gate'):
            if hasattr(self.memory, name):
                delattr(self.memory, name)
