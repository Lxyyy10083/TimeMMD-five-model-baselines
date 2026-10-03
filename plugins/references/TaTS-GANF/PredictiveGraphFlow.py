"""Backbone-independent relevance selection and graph-conditioned future density."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class GraphAffine(nn.Module):
    """Triangular flow: each coordinate sees only its ordered graph parents.

    Edges describe dependencies between fixed future trajectory coefficients,
    not causal relationships between real-world variables.
    """
    def __init__(self, size, width, reverse=False):
        super().__init__()
        order = torch.arange(size)
        if reverse:
            order = order.flip(0)
        self.register_buffer('order', order)
        rank = order.argsort()
        self.register_buffer('mask', (rank[:, None] > rank[None, :]).float())
        self.edges = nn.Parameter(torch.full((size, size), -2.0))
        self.condition_edges = nn.Linear(width, size)
        self.coordinate = nn.Embedding(size, width)
        self.parameters_net = nn.Sequential(
            nn.Linear(width + size, width), nn.SiLU(), nn.Linear(width, 2))
        nn.init.zeros_(self.parameters_net[-1].weight)
        nn.init.zeros_(self.parameters_net[-1].bias)

    def adjacency(self, context):
        # Row=child, column=parent. A strict ordering guarantees acyclicity.
        strength = torch.sigmoid(self.condition_edges(context))
        return torch.sigmoid(self.edges)[None] * self.mask * strength[:, None, :]

    def affine_parameters(self, x, context, adjacency):
        parents = adjacency * x[:, None, :]
        cond = context[:, None, :] + self.coordinate.weight[None]
        shift, raw_scale = self.parameters_net(torch.cat([cond, parents], -1)).unbind(-1)
        return shift, 0.7 * torch.tanh(raw_scale)

    def forward(self, x, context):
        shift, log_scale = self.affine_parameters(x, context, self.adjacency(context))
        return (x - shift) * torch.exp(-log_scale), -log_scale.sum(-1)

    def inverse(self, z, context):
        x = torch.zeros_like(z)
        adjacency = self.adjacency(context)
        for coordinate in self.order.unbind():
            shift, log_scale = self.affine_parameters(x, context, adjacency)
            value = z[:, coordinate] * log_scale[:, coordinate].exp() + shift[:, coordinate]
            # Out-of-place update preserves the sampling path's autograd graph.
            selector = F.one_hot(coordinate, z.shape[-1]).to(z.dtype)
            x = x + value[:, None] * selector
        return x


class ConditionalGraphFlow(nn.Module):
    def __init__(self, size, width, blocks, samples):
        super().__init__()
        self.layers = nn.ModuleList([GraphAffine(size, width, bool(i % 2))
                                     for i in range(blocks)])
        # Fixed antithetic normal draws: reproducible train/eval moments,
        # no evaluation-time RNG or dependence on evaluation batch size.
        engine = torch.quasirandom.SobolEngine(size, scramble=True, seed=1729)
        uniform = engine.draw(max(1, samples // 2)).clamp(1e-5, 1 - 1e-5)
        z = math.sqrt(2) * torch.erfinv(2 * uniform - 1)
        self.register_buffer('noise', torch.cat([z, -z], 0))

    def nll(self, target, context):
        z, logdet = target, target.new_zeros(target.shape[0])
        for layer in self.layers:
            z, increment = layer(z, context)
            logdet = logdet + increment
        return (0.5 * (z.square() + math.log(2 * math.pi)).sum(-1) - logdet).mean() / target.shape[-1]

    def samples(self, context):
        batch, width = context.shape
        count, size = self.noise.shape
        cond = context[:, None].expand(batch, count, width).reshape(-1, width)
        z = self.noise[None].expand(batch, count, size).reshape(-1, size)
        for layer in reversed(self.layers):
            z = layer.inverse(z, cond)
        return z.reshape(batch, count, size)

    def sparsity(self):
        return torch.stack([(torch.sigmoid(layer.edges) * layer.mask).sum()
                            / layer.mask.sum().clamp_min(1) for layer in self.layers]).mean()


class PredictiveGraphFlow(nn.Module):
    """Full TaTS process around any [B,L,1] -> [B,H,1] forecasting backbone."""
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        self.horizon = args.pred_len
        self.label_len = args.label_len
        width, size = args.full_width, args.full_features
        self.text_projection = nn.Sequential(
            nn.Linear(args.llm_dim, width), nn.LayerNorm(width), nn.GELU(),
            nn.Dropout(args.full_dropout))
        self.history = nn.GRU(1, width, batch_first=True)
        self.relevance = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))
        self.channel_gate = nn.Linear(width, width)
        self.refinement = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.refinement.weight)
        self.context = nn.Sequential(nn.Linear(2 * width + size, width), nn.LayerNorm(width), nn.SiLU())
        self.ts_probe = nn.Linear(width, size)
        self.text_probe = nn.Linear(width, size, bias=False)
        self.flow = ConditionalGraphFlow(size, width, args.full_blocks, args.full_samples)
        self.correction_logit = nn.Parameter(torch.tensor(-2.0))
        # Fixed orthonormal DCT basis; coeffs divided by sqrt(H) for comparable
        # density scale across prediction horizons. No learned target encoder.
        t = torch.arange(self.horizon).float()[:, None] + 0.5
        k = torch.arange(size).float()[None, :]
        basis = torch.cos(math.pi * t * k / self.horizon) * math.sqrt(2 / self.horizon)
        basis[:, 0] /= math.sqrt(2)
        self.register_buffer('basis', basis)

    def coefficients(self, trajectory):
        return trajectory.squeeze(-1) @ self.basis / math.sqrt(self.horizon)

    def forward(self, x, text, text_valid, x_mark, y_mark):
        location = x.mean(1, keepdim=True).detach()
        scale = x.std(1, keepdim=True, unbiased=False).detach().clamp_min(0.1)
        history, _ = self.history((x - location) / scale)
        ts = history[:, -1]
        text = self.text_projection(text)
        logits = self.relevance(torch.cat([history, text, history * text], -1))
        gate = logits.sigmoid() * text_valid.unsqueeze(-1)
        selected = gate * text * self.channel_gate(ts).sigmoid()[:, None]
        summary = selected.mean(1)
        refined = x + 0.1 * scale * torch.tanh(self.refinement(selected))
        decoder = torch.cat([refined[:, -self.label_len:] if self.label_len else refined[:, :0],
                             x.new_zeros(x.shape[0], self.horizon, 1)], 1)
        base = self.backbone(refined, x_mark, decoder, y_mark)
        if isinstance(base, tuple):
            base = base[0]
        base = base[:, -self.horizon:, :1]
        context = self.context(torch.cat([ts, summary,
                               self.coefficients((base.detach() - location) / scale)], -1))
        # Flow arithmetic remains float32, including under automatic mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            draws = self.flow.samples(context.float())
            mean = draws.mean(1)
            trajectories = math.sqrt(self.horizon) * (draws @ self.basis.T)
            variance = trajectories.var(1, unbiased=False)
            gain = self.correction_logit.sigmoid() / (1 + variance)
            prediction = base.float() + scale * (gain * trajectories.mean(1)).unsqueeze(-1)
        return prediction, dict(base=base, context=context, mean=mean, scale=scale,
                                gate=gate, logits=logits, valid=text_valid,
                                probe_ts=self.ts_probe(ts), probe_text=self.text_probe(summary),
                                last=x[:, -1:], variance=variance, gain=gain)

    def objective(self, prediction, target, state, args, warmup):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            residual = self.coefficients((target.float() - state['base'].detach().float()) / state['scale'])
            nll = self.flow.nll(residual, state['context'].float())
            feature_target = self.coefficients((target.float() - state['last']) / state['scale'])
            ts_probe = state['probe_ts'].float()
            joint_probe = ts_probe + state['probe_text'].float()
            # Supervise usefulness by forecast error reduction, not text/TS similarity.
            error_ts = (ts_probe.detach() - feature_target).square().mean(-1)
            error_joint = (joint_probe.detach() - feature_target).square().mean(-1)
            utility = ((error_ts - error_joint) / (0.1 + error_ts)).clamp(-1, 1)
            # Demand a positive margin: no measurable gain should favor closing
            # the gate, rather than receiving a neutral target of 0.5.
            useful = torch.sigmoid((utility - 0.05) / 0.025).detach()
            valid = state['valid'].float()
            relevance = F.binary_cross_entropy_with_logits(
                state['logits'].squeeze(-1).float(), useful[:, None].expand_as(valid), reduction='none')
            relevance = (relevance * valid).sum() / valid.sum().clamp_min(1)
            probes = F.mse_loss(ts_probe, feature_target) + F.mse_loss(joint_probe, feature_target)
            mse = F.mse_loss(prediction, target)
            moment = F.mse_loss(state['mean'], residual)
            loss = (mse + warmup * args.full_nll_weight * nll
                    + args.full_alignment_weight * (moment + probes)
                    + args.full_relevance_weight * relevance
                    + args.full_sparse_weight * (state['gate'].mean() + self.flow.sparsity()))
        return loss, dict(mse=mse.detach(), nll=nll.detach(), relevance=relevance.detach())
