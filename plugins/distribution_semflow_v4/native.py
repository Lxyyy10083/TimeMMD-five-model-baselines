"""Native backbones with differentiable internal integration.

Every process loads exactly one upstream repository to prevent 'models/layers'
namespace collisions. Original source files are never changed.
"""
from __future__ import annotations
import argparse
import ast
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from .core import InternalFusion, SemanticDistribution

ROOT = Path(__file__).resolve().parents[2]
REPOS = {'SpecTF': 'SpecTF-main', 'CFA': 'cfa-main', 'TaTS': 'TaTS-main',
         'MM-TSFlib': 'MM-TSFlib-main', 'Aurora': 'Aurora-main/TimeMMD'}


def config_from_upstream(model, domain, horizon, seq_len, seed):
    """Read only declarative parser statements, without executing upstream runs."""
    if model == 'Aurora':
        return None
    sys.path.insert(0, str(ROOT / 'benchmark'))
    import run_all
    run_all.DATA = ROOT / 'benchmark/readgpt_data'
    _, command = run_all.command(model, domain, horizon, seq_len, 200, seed)
    tree = ast.parse((ROOT / REPOS[model] / 'run.py').read_text(encoding='utf-8'))
    main = next(s for s in tree.body if isinstance(s, ast.If)
                and isinstance(s.test, ast.Compare)
                and isinstance(s.test.left, ast.Name) and s.test.left.id == '__name__')
    statements = []
    for s in main.body:
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in s.targets):
            statements.append(s)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            f = s.value.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'parser':
                statements.append(s)
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=statements, type_ignores=[]), 'upstream_parser', 'exec'), scope)
    cfg = scope['parser'].parse_args(command[3:])
    cfg.llm_dim = 768
    cfg.enc_in = 1 + cfg.text_emb if model == 'TaTS' else 1
    cfg.dec_in = cfg.enc_in
    cfg.c_out = 1
    if model in ('MM-TSFlib', 'CFA'):
        cfg.text_emb = horizon
    cfg.output_attention = False
    cfg.use_gpu = torch.cuda.is_available()
    cfg.use_amp = False
    if model == 'TaTS':
        cfg.prompt_weight = cfg.prior_weight
    return cfg


class NativeForecast(nn.Module):
    def __init__(self, model, config, horizon, variant='internal', width=64):
        super().__init__()
        self.name, self.config, self.horizon = model, config, horizon
        self.variant = variant
        self.has_distribution = variant not in ('control', 'fusion_only')
        self.has_fusion = variant not in ('control', 'distribution_only')
        # Adding a repository once is safe because each experiment is a process.
        sys.path.insert(0, str(ROOT / REPOS[model]))
        self.fusions = nn.ModuleList()
        self.handles = []
        self._last_hidden = None
        if model == 'SpecTF':
            from models.SpecTF import TextEncoder
            from .spectral import DistributionSpecTF
            self.backbone = DistributionSpecTF(config)
            self.backbone.use_all_bands = False
            self.text_encoder = TextEncoder(config)
            feature_dim = 2 * config.mm_emb_size
            if self.has_fusion:
                self.fusions.extend([InternalFusion(config.mm_emb_size, width, True),
                                     InternalFusion(config.mm_emb_size, width, True)])
            self.backbone.internal_fuse = self._spectral_fuse
            # Always capture the complex future hidden before its scalar projection.
            self.handles.append(self.backbone.register_forward_hook(self._assert_hidden))
        elif model == 'CFA':
            from models.PatchTST import Model
            self.backbone = Model(config)
            self.text_encoder = nn.Sequential(nn.Linear(768, 96), nn.ReLU(),
                                              nn.Dropout(0.3), nn.Linear(96, config.text_emb))
            feature_dim = config.d_model
            self._install_layers(self.backbone.encoder.attn_layers, feature_dim, width)
        elif model == 'TaTS':
            from models.iTransformer import Model
            self.backbone = Model(config)
            self.text_encoder = nn.Sequential(nn.Linear(768, 96), nn.ReLU(),
                                              nn.Dropout(0.3), nn.Linear(96, config.text_emb))
            feature_dim = config.d_model
            self._install_layers(self.backbone.encoder.attn_layers, feature_dim, width)
        elif model == 'MM-TSFlib':
            from models.Informer import Model
            self.backbone = Model(config)
            # Text-to-horizon branch mirrors the native output-blending family.
            self.text_encoder = nn.Sequential(nn.Linear(768, 96), nn.ReLU(),
                                              nn.Dropout(0.3), nn.Linear(96, horizon))
            feature_dim = config.d_model
            self._install_layers(self.backbone.encoder.attn_layers, feature_dim, width)
            self._install_layers(self.backbone.decoder.layers, feature_dim, width)
            self.handles.append(self.backbone.decoder.projection.register_forward_pre_hook(
                self._capture_projection))
        elif model == 'Aurora':
            from aurora.modeling_aurora import AuroraForPrediction
            self.backbone = AuroraForPrediction.from_pretrained(str(ROOT / 'models/aurora'))
            # The upstream inner AuroraModel omits this attribute in __init__;
            # direct differentiable calls still read it for default flags.
            self.backbone.model.config = self.backbone.config
            self.backbone.requires_grad_(False)
            # Fine-tune the last encoder/decoder blocks and native point head.
            for block in list(self.backbone.model.enc_layers)[-2:] + list(self.backbone.model.dec_layers)[-2:]:
                block.requires_grad_(True)
            self.backbone.linear_head.requires_grad_(True)
            feature_dim = self.backbone.config.hidden_size
            self._install_layers(self.backbone.model.dec_layers, feature_dim, width)
            self.token_len = self.backbone.config.token_len
            self.register_buffer('token_identity', torch.eye(self.token_len))
        else:
            raise ValueError(model)
        self.distribution = SemanticDistribution(horizon, feature_dim, width)
        if variant == 'control':
            self.distribution.requires_grad_(False)
        elif not self.has_distribution:
            for module in (self.distribution.hidden_projection, self.distribution.context,
                           self.distribution.flow, self.distribution.reliability, self.distribution.forecast_projection):
                module.requires_grad_(False)
        if False:  # V3 preserves trainable native CFA text adapters.
            self.text_encoder.requires_grad_(False)
            for layer in self.backbone.encoder.attn_layers:
                for name in ('adapter_down', 'adapter_up', 'adapter_norm'):
                    if hasattr(layer, name):
                        getattr(layer, name).requires_grad_(False)

    def _install_layers(self, layers, feature_dim, width):
        for layer in layers:
            index = len(self.fusions)
            if self.has_fusion:
                self.fusions.append(InternalFusion(feature_dim, width))
            def hook(module, inputs, output, slot=index):
                hidden = output[0] if isinstance(output, tuple) else output
                if self.has_fusion:
                    hidden = self.fusions[slot](hidden, self.distribution.memory)
                self._last_hidden = hidden
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            self.handles.append(layer.register_forward_hook(hook))

    def _capture_projection(self, module, inputs):
        self._last_hidden = inputs[0]

    def _capture_spectrum(self, module, inputs, output):
        # Capture source spectral features without detaching. Their last axis is F.
        if not self.has_fusion:
            incoming = inputs[0].permute(0, 1, 3, 2)
            self._last_hidden = torch.cat((incoming, incoming.new_zeros(incoming.shape)), -1)

    def _assert_hidden(self, module, inputs, output):
        if self._last_hidden is None:
            raise RuntimeError('SpecTF internal hidden capture did not execute')

    def _spectral_fuse(self, features, stage):
        result = self.fusions[stage](features, self.distribution.memory) if self.has_fusion else features
        self._last_hidden = result
        return result

    def train(self, mode=True):
        super().train(mode)
        if self.name == 'Aurora':
            # Disable native masked augmentation and retain identical tokenization
            # in training/evaluation; eval does not disable autograd.
            self.backbone.eval()
        return self

    def forward(self, batch):
        x, text, mask = batch['history'], batch['text'], batch['mask']
        if self.variant == 'numeric':
            text, mask = torch.zeros_like(text), torch.zeros_like(mask)
        elif self.variant == 'shuffle':
            # Only reorder already-observed text within the current history.
            order = torch.arange(text.shape[1] - 1, -1, -1, device=text.device)
            text, mask = text[:, order], mask[:, order]
        self.distribution.memory(x, text, mask)
        self._last_hidden = None
        if self.name == 'Aurora':
            x2 = x.squeeze(-1)
            mean = x2.mean(1, keepdim=True).detach()
            scale = x2.std(1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
            token_count = math.ceil(self.horizon / self.token_len)
            # This path does not call generate or no_grad; decoder gets gradients.
            use_text = self.variant != 'numeric'
            result = self.backbone.model(input_ids=(x2 - mean) / scale,
                text_input_ids=batch['input_ids'] if use_text else None,
                text_attention_mask=batch['attention_mask'] if use_text else None,
                text_token_type_ids=batch['token_type_ids'] if use_text else None, predict_token_num=token_count,
                inference_token_len=self.token_len, return_dict=True)
            hidden = result.last_hidden_state[1]
            self._last_hidden = hidden
            normalized = self.backbone.linear_head.output_proj(hidden).flatten(1)[:, :self.horizon]
            native = (normalized * scale + mean)[..., None]
        else:
            cfg = self.config
            marks = batch['marks']
            future_marks = batch['future_marks']
            dec = torch.cat((x[:, -cfg.label_len:] if cfg.label_len else x[:, :0],
                             x.new_zeros(len(x), self.horizon, 1)), 1)
            if self.name == 'SpecTF':
                encoded = self.text_encoder(text * mask[..., None], marks)
                encoded = encoded * mask[:, None, :, None, None]
                native = self.backbone(x, marks, None, None, encoded)
            elif self.name == 'CFA':
                encoded = self.text_encoder(text) * mask[..., None]
                pooled = encoded.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                # Replace repeated pooled CFA injection with token-specific lookup.
                context = pooled
                native = self.backbone(x, marks, dec, future_marks, text_context=context)
            elif self.name == 'TaTS':
                encoded = self.text_encoder(text) * mask[..., None]
                full = torch.cat((x, encoded), -1)
                full_dec = torch.cat((dec, dec.new_zeros(len(x), dec.shape[1], cfg.text_emb)), -1)
                native = self.backbone(full, marks, full_dec, future_marks)[:, -self.horizon:, :1]
            else:
                native = self.backbone(x, marks, dec, future_marks)[:, -self.horizon:, :1]
                if cfg.prompt_weight:
                    encoded = self.text_encoder(text) * mask[..., None]
                    pooled = encoded.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                    pooled = (pooled - pooled.mean(1, keepdim=True)) / pooled.std(
                        1, keepdim=True, unbiased=False).clamp_min(1e-5)
                    native = (1 - cfg.prompt_weight) * native + cfg.prompt_weight * (
                        pooled[..., None] + batch['prior'])
            if self.name in ('SpecTF', 'TaTS'):
                native = (1 - cfg.prompt_weight) * native + cfg.prompt_weight * batch['prior']
        if self._last_hidden is None:
            raise RuntimeError(f'{self.name}: hidden feature hook failed')
        if self.has_distribution:
            return self.distribution.finish(native, self._last_hidden)
        self.distribution.state = {'base': native, 'prediction': native}
        return native

    def objective(self, target, **weights):
        if self.has_distribution:
            return self.distribution.objective(target, **weights)
        error = self.distribution.state['prediction'] - target
        value = error.square().mean() + weights.get('mae_weight', 0.5) * error.abs().mean()
        return value, dict(point=float(value.detach()))
