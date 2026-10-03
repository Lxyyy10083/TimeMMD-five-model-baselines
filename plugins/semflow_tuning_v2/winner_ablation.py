"""Validate text dependence of the locked winner without selecting on test data."""
from pathlib import Path
import json
import sys
import torch

from experiment import (OUTPUT, PILOT, manifests, text_cache, variants, train_case,
                        ranking, write_json)
import argparse


def main():
    torch.set_num_threads(4)
    p=argparse.ArgumentParser()
    p.add_argument('--bert',default='/root/autodl-tmp/carma_workspace/models/bert-base-uncased')
    args=p.parse_args()
    args.batch_size=256;args.lr=0.001;args.screen_epochs=24;args.final_epochs=40
    args.minimum_epochs=8;args.patience=8
    winner=json.loads((OUTPUT/'winner.json').read_text())['variant']
    cfg=variants()[winner]
    ablations={'winner_numeric':{**cfg,'text_mode':'none'},
               'winner_shuffle':{**cfg,'text_mode':'shuffle'}}
    for meta in manifests():
        domain=meta['domain'];horizon=PILOT[domain]
        embedding,mask=text_cache(domain,Path(args.bert))
        for seed in (2026,2027):
            for name,config in ablations.items():
                train_case('ablation',name,config,domain,horizon,seed,args,embedding,mask)
    scores=ranking('ablation',list(ablations),[2026,2027])
    real=next(row for row in json.loads((OUTPUT/'confirmation_ranking.json').read_text())
              if row['variant']==winner)
    write_json(OUTPUT/'winner_ablation.json',dict(winner=winner,real_text=real,ablations=scores,
                  note='Holdout comparison only; locked winner and test results are not changed.'))
    print('WINNER_ABLATION_COMPLETED',flush=True)


if __name__=='__main__':main()
