"""Serial experiment driver; all trials save validation only until --stage evaluate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ['SpecTF', 'CFA', 'TaTS', 'MM-TSFlib', 'Aurora']
PILOT = [('Agriculture', 12), ('Economy', 8), ('Environment', 192), ('Security', 12)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['pilot', 'full'], default='pilot')
    p.add_argument('--stage', choices=['train', 'evaluate'], default='train')
    p.add_argument('--output', default=str(HERE/'outputs_v1'))
    p.add_argument('--models', nargs='+', choices=MODELS, default=['SpecTF','CFA'])
    p.add_argument('--variants', nargs='+', default=['control','internal','numeric',
                                                   'fusion_only','distribution_only','shuffle'])
    p.add_argument('--seeds', nargs='+', type=int, default=[2026])
    p.add_argument('--epochs', type=int, default=200)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'evaluate' and not (out/'DESIGN_LOCKED.json').exists():
        p.error('final evaluation requires DESIGN_LOCKED.json recording holdout selection')
    if args.suite == 'full':
        tasks = []
        for file in sorted((ROOT/'benchmark/readgpt_data').glob('*/manifest.json')):
            info = json.loads(file.read_text(encoding='utf-8'))
            tasks.extend((info['domain'], h) for h in info['horizons'])
        if not tasks:
            raise ValueError('canonical manifests are unavailable')
    else:
        tasks = PILOT
    plan = dict(arguments=vars(args), tasks=tasks, jobs=len(tasks)*len(args.models)*
                len(args.variants)*len(args.seeds))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    (out/f'launch_{stamp}.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               MPLBACKEND='Agg')
    failures = []
    for model in args.models:
        for domain, horizon in tasks:
            for seed in args.seeds:
                for variant in args.variants:
                    dest = out/variant/model/domain/str(horizon)/str(seed)
                    done = dest/('COMPLETED.json' if args.stage=='train' else 'EVALUATED.json')
                    if done.exists():
                        continue
                    log = out/'logs'/f'{args.stage}_{variant}_{model}_{domain}_{horizon}_{seed}.log'
                    log.parent.mkdir(exist_ok=True)
                    cmd = [sys.executable, '-u', str(HERE/'train.py'), '--model', model,
                        '--domain', domain, '--horizon', str(horizon), '--variant', variant,
                        '--seed', str(seed), '--epochs', str(args.epochs),
                        '--stage', args.stage, '--output', str(out)]
                    print('START', model, domain, horizon, seed, variant, flush=True)
                    with log.open('a', encoding='utf-8') as stream:
                        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                                stderr=subprocess.STDOUT)
                    if result.returncode:
                        failures.append(dict(model=model, domain=domain, horizon=horizon,
                                             seed=seed, variant=variant, log=str(log)))
                        # An integration exception likely affects many tasks.
                        (out/'FAILED.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
                        print('FAILED', str(log), flush=True)
                        return 1
                    print('DONE', model, domain, horizon, seed, variant, flush=True)
    (out/f'SUITE_{args.stage}_COMPLETED.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())

