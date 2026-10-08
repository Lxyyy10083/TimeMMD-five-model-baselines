"""CPU验证真实原模型接入、梯度、参数更新及条件流可逆性；不运行正式实验。

Aurora验证使用原生encoder/decoder/head，冻结视觉/文本资产用小型测试替身，
不加载或验证正式预训练权重。每个原仓库在独立子进程导入。
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from unittest.mock import patch
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'plugins'))
from plugins.v1_joint_internal_20261008.native import JointForecast, config_from_upstream
from plugins.v1_joint_internal_20261008.core import JointDistribution, TrajectoryFlow


def grad_total(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def make_aurora():
    sys.path.insert(0, str(ROOT / 'Aurora-main/TimeMMD'))
    from aurora.configuration_aurora import AuroraConfig
    import aurora.modeling_aurora as native

    class Vision(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.projection = nn.Linear(1, config.hidden_size)
            self.tokens = config.num_distill

        def forward(self, values, **kwargs):
            return self.projection(values.mean(1)[:, None, None]).expand(-1, self.tokens, -1)

    class Text(Vision):
        def forward(self, tokens):
            return super().forward(tokens['input_ids'].float())

    config = AuroraConfig(token_len=6, hidden_size=16, intermediate_size=32, num_enc_layers=2,
        num_dec_layers=2, num_attention_heads=2, dropout_rate=0, num_distill=2,
        num_prototypes=4, flow_loss_depth=1)
    with patch.object(native, 'VisionEncoder', Vision), patch.object(native, 'TextEncoder', Text):
        backbone = native.AuroraForPrediction(config)
    with patch.object(native.AuroraForPrediction, 'from_pretrained', return_value=backbone):
        return JointForecast('Aurora', None, 6, width=8, samples=4)


def check_model(name):
    torch.set_num_threads(2); torch.manual_seed(7)
    if name == 'Aurora':
        model = make_aurora()
        config = None
    else:
        config = config_from_upstream(name, 'Agriculture', 6, 24, 2026)
        config.d_model = 16; config.d_ff = 32; config.n_heads = 2
        config.e_layers = 1; config.d_layers = 1; config.dropout = 0
        model = JointForecast(name, config, 6, width=8, samples=4)
    batch = dict(history=torch.randn(2, 24, 1), text=torch.randn(2, 24, 768), mask=torch.ones(2, 24),
        marks=torch.zeros(2, 24, 1), future_marks=torch.zeros(2, 18 if name != 'SpecTF' else 6, 1),
        prior=torch.zeros(2, 6, 1), input_ids=torch.ones(2, 12, dtype=torch.long),
        attention_mask=torch.ones(2, 12, dtype=torch.long), token_type_ids=torch.zeros(2, 12, dtype=torch.long))
    target = torch.randn(2, 6, 1)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    before_native = {n: p.detach().clone() for n, p in model.backbone.named_parameters() if p.requires_grad}
    before_flow = {n: p.detach().clone() for n, p in model.distribution.flow.named_parameters()}
    pred = model(batch)
    assert pred.shape == target.shape and torch.isfinite(pred).all()
    loss, _ = model.objective(target)
    loss.backward()
    assert grad_total(model.backbone) > 0 and grad_total(model.fusions) > 0 and grad_total(model.distribution.flow) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step(); model.distribution.release()
    assert any(not torch.equal(p, before_native[n]) for n, p in model.backbone.named_parameters() if n in before_native)
    assert any(not torch.equal(p, before_flow[n]) for n, p in model.distribution.flow.named_parameters())
    # 独立验证NLL更新主干和内部适配器，不仅是总loss中点预测项产生梯度。
    optimizer.zero_grad(set_to_none=True); model(batch)
    model.distribution.density(target)[0].mean().backward()
    nll_backbone = grad_total(model.backbone)
    nll_fusion = grad_total(model.fusions)
    assert nll_backbone > 0 and nll_fusion > 0
    model.distribution.release()
    # 独立验证点预测loss直接更新flow；避免V3曾有的独立位置头问题。
    optimizer.zero_grad(set_to_none=True); pred = model(batch)
    (pred - target).square().mean().backward()
    point_flow = grad_total(model.distribution.flow)
    assert point_flow > 0
    model.distribution.release()
    # 缺失全部文本时混合门控精确为0，密度和预测仍有限。
    missing = {**batch, 'mask': torch.zeros_like(batch['mask']), 'text': torch.full_like(batch['text'], float('nan'))}
    pred = model(missing)
    assert torch.isfinite(pred).all() and torch.count_nonzero(model.distribution.state['gate']) == 0
    assert torch.isfinite(model.distribution.density(target)[0]).all()
    model.distribution.release()
    print(json.dumps(dict(model=name, native_layers='passed', native_and_flow_parameters_updated=True,
        nll_backbone_gradient=nll_backbone, nll_adapter_gradient=nll_fusion, point_flow_gradient=point_flow,
        aurora_assets='test substitutes; original transformer layers' if name == 'Aurora' else None)))


def check_flow():
    torch.set_num_threads(2); torch.manual_seed(8)
    for horizon in (1, 6, 336):
        flow = TrajectoryFlow(horizon, width=4, layers=4, samples=4)
        for layer in flow.layers:
            nn.init.normal_(layer.out[-1].weight, std=0.02)
            nn.init.normal_(layer.out[-1].bias, std=0.02)
        context, values = torch.randn(2, horizon, 4), torch.randn(2, horizon)
        z = values
        for layer in flow.layers:
            z, _ = layer.normalize(z, context)
        reconstructed = z
        for layer in reversed(flow.layers):
            reconstructed = layer.inverse(reconstructed, context)
        assert torch.allclose(values, reconstructed, atol=1e-5, rtol=1e-5)
        assert torch.isfinite(flow.nll(values, context)).all()
    module = JointDistribution(1, feature_dim=8, width=4, samples=4)
    module.memory(torch.randn(2, 24, 1), torch.full((2, 24, 768), float('nan')), torch.zeros(2, 24))
    prediction = module.finish(torch.randn(2, 1, 1), torch.randn(2, 3, 8))
    assert prediction.shape == (2, 1, 1) and torch.isfinite(prediction).all()
    loss, _ = module.objective(torch.randn_like(prediction)); loss.backward()
    assert grad_total(module.flow) > 0
    print(json.dumps(dict(flow_roundtrip_horizons=[1, 6, 336], one_step_density='passed', missing_text='passed')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model'); args = parser.parse_args()
    if args.model:
        check_model(args.model)
    else:
        check_flow()
        for name in ('TaTS', 'CFA', 'MM-TSFlib', 'SpecTF', 'Aurora'):
            subprocess.run([sys.executable, str(HERE / 'verify.py'), '--model', name], check=True)
