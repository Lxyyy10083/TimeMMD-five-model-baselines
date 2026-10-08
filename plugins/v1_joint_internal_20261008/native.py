"""五模型内部接入；每个进程只导入一个原模型，避免models/layers名称冲突。

复用已有内部实验的原生模型加载与输入适配代码，替换内部适配器和分布头。
原生输入仍采用该加载器的BERT协议，不能声称逐行复现历史GPT2输入。
"""
from __future__ import annotations
import torch
from ..internal_semflow_full_lab.native import NativeForecast, config_from_upstream, REPOS
from .core import InternalAdapter, JointDistribution


class JointForecast(NativeForecast):
    def __init__(self, model, config, horizon, variant='internal', width=32, samples=16):
        if variant not in ('internal', 'control'):
            raise ValueError('variant must be internal or control')
        self._joint_enabled = variant == 'internal'
        # 使用原生加载分支，不使用旧实验的SemanticDistribution或旧InternalFusion。
        # _install_layers在父类构造时调用本类覆写，实际注册下方的可微内部模块。
        super().__init__(model, config, horizon, variant='control', width=width)
        if model == 'Aurora':
            # 【修改Aurora训练范围】时序编码器/解码器及原生融合、预测头均可训练。
            # 仅保留大型视觉/文本预训练编码器冻结；旧flow_match/retriever未用于本路径。
            for name, module in self.backbone.model.named_children():
                if name not in ('VisionEncoder', 'TextEncoder'):
                    module.requires_grad_(True)
            self.backbone.model.W.requires_grad_(True)
            self.backbone.linear_head.requires_grad_(True)
        feature_dim = (2 * config.mm_emb_size if model == 'SpecTF' else
                       self.backbone.config.hidden_size if model == 'Aurora' else config.d_model)
        self.distribution = JointDistribution(horizon, feature_dim, width, samples)
        self.variant = variant
        self.has_distribution = self._joint_enabled
        self.has_fusion = self._joint_enabled
        if model == 'SpecTF' and self._joint_enabled:
            # 【接入SpecTF】历史/未来复数频谱，原解码标量投影与irfft之前。
            self.fusions.extend([InternalAdapter(config.mm_emb_size, width, True) for _ in range(2)])
        if not self._joint_enabled:
            self.distribution.requires_grad_(False)

    def _install_layers(self, layers, feature_dim, width):
        # 【接入TaTS/CFA】encoder各层输出，原预测头之前；CFA原生adapter保留。
        # 【接入MM-TSFlib】encoder与decoder各层输出，原projection之前。
        # 【接入Aurora】decoder各层输出，linear_head之前；时序主干参与联合训练。
        for layer in layers:
            slot = len(self.fusions)
            if self._joint_enabled:
                self.fusions.append(InternalAdapter(feature_dim, width))

            def hook(module, inputs, output, index=slot):
                hidden = output[0] if isinstance(output, tuple) else output
                if self._joint_enabled:
                    hidden = self.fusions[index](hidden, self.distribution.memory)
                self._last_hidden = hidden
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            self.handles.append(layer.register_forward_hook(hook))

    def _spectral_fuse(self, features, stage):
        result = self.fusions[stage](features, self.distribution.memory) if self._joint_enabled else features
        self._last_hidden = result
        return result

    def forward(self, batch):
        # 原生文本分支也屏蔽缺失值，避免NaN*0污染隐藏表示。
        text = torch.where(batch['mask'][..., None] > 0, batch['text'], torch.zeros_like(batch['text']))
        return super().forward({**batch, 'text': text})

    def train(self, mode=True):
        super().train(mode)
        if self.name == 'Aurora':
            # 顶层AuroraModel保持eval以禁用原生掩码扩增（会改变batch大小）；
            # 时序子层恢复训练模式与dropout，eval标志不会关闭autograd。
            for name, module in self.backbone.model.named_children():
                if name not in ('VisionEncoder', 'TextEncoder'):
                    module.train(mode)
        return self

    def objective(self, target, **weights):
        if self.has_distribution:
            return self.distribution.objective(target, **weights)
        error = self.distribution.state['prediction'] - target
        point = error.square().mean() + weights.get('mae_weight', 0.2) * error.abs().mean()
        return point, dict(point=float(point.detach()))
