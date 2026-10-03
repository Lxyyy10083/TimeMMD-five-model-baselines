"""Patch only the experiment copy to export ordered validation predictions.

Run once from the repository root. Source baselines remain available in Git.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = {
    'MM-TSFlib-main': 'exp/exp_long_term_forecasting.py',
    'cfa-main': 'exp/exp_long_term_forecasting_text_integrated.py',
    'SpecTF-main': 'exp/exp_long_term_forecasting.py',
    'TaTS-main': 'exp/exp_long_term_forecasting.py',
}

for repo, rel in paths.items():
    path = ROOT / repo / rel
    source = path.read_text(encoding='utf-8')
    pos = source.index('    def test(self, setting')
    prefix, body = source[:pos], source[pos:]
    old = "test_data, test_loader = self._get_data(flag='test')"
    assert body.count(old) == 1, (path, body.count(old))
    body = body.replace(old, "split = os.environ.get('CARMA_EXPORT_SPLIT', 'test')\n        test_data, test_loader = self._get_data(flag=split)", 1)
    body = body.replace("folder_path = './test_results/' + setting + '/'", "folder_path = './' + ('val_test_results' if split == 'val' else 'test_results') + '/' + setting + '/'", 1)
    body = body.replace("folder_path = './results/' + setting + '/'", "folder_path = './' + ('val_results' if split == 'val' else 'results') + '/' + setting + '/'", 1)
    if repo == 'cfa-main':
        body = body.replace("#         np.save(folder_path + 'pred.npy', preds)", "        np.save(folder_path + 'pred.npy', preds)", 1)
        body = body.replace("#         np.save(folder_path + 'true.npy', trues)", "        np.save(folder_path + 'true.npy', trues)", 1)
        body = body.replace("if os.path.exists(checkpoint_path):", "if split == 'test' and not os.environ.get('CARMA_KEEP_CHECKPOINTS') and os.path.exists(checkpoint_path):", 1)
    path.write_text(prefix + body, encoding='utf-8')

    factory = ROOT / repo / 'data_provider' / 'data_factory.py'
    f = factory.read_text(encoding='utf-8')
    assert "shuffle_flag = False if flag == 'test' else True" in f
    f = f.replace("shuffle_flag = False if flag == 'test' else True", "shuffle_flag = False if flag in ('test', 'val') else True", 1)
    factory.write_text(f, encoding='utf-8')

aurora = ROOT / 'Aurora-main/TimeMMD/exp/exp_main.py'
source = aurora.read_text(encoding='utf-8')
pos = source.index('    def test(self, setting):')
prefix, body = source[:pos], source[pos:]
body = body.replace("test_data, test_loader = self._get_data(flag='test')", "split = os.environ.get('CARMA_EXPORT_SPLIT', 'test')\n        test_data, test_loader = self._get_data(flag=split)", 1)
needle = "            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))"
assert needle in body
body = body.replace(needle, needle + "\n            folder = './' + ('val_results' if split == 'val' else 'results') + '/' + setting + '/'\n            os.makedirs(folder, exist_ok=True)\n            np.save(folder + 'pred.npy', preds)\n            np.save(folder + 'true.npy', trues)", 1)
aurora.write_text(prefix + body, encoding='utf-8')

factory = ROOT / 'Aurora-main/TimeMMD/data_provider/data_factory.py'
f = factory.read_text(encoding='utf-8')
assert "shuffle_flag = False if flag == 'test' else True" in f
factory.write_text(f.replace("shuffle_flag = False if flag == 'test' else True", "shuffle_flag = False if flag in ('test', 'val') else True", 1), encoding='utf-8')
print('Patched validation export in experiment copy only')
