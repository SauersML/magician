"""Runs chosen held-out instructions with the trained model's own plan, the original model's plan and the exact
plan, each placed in the thinking block, and saves the achieved and requested changes per reply token and layer for
figures/make_figures.py (plan figure).

Usage: python examples.py --adapter A --eval results/eval_<arm>.json --planner_eval results/eval_<arm>think.json
           --set new_layer_patterns --items 94,66 --out results/plan_examples_<arm>.pt
The plans are the ones stored in the two evaluation files, so the own-plan scores reproduce eval_<arm>.json.
"""
import argparse
import json
import os

import numpy as np
import torch
from peft import PeftModel

import common as C
import measure
import spec

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--eval', required=True)
ap.add_argument('--planner_eval', required=True)
ap.add_argument('--set', default='new_layer_patterns')
ap.add_argument('--items', required=True, help='comma-separated item indices in the evaluation set')
ap.add_argument('--out', required=True)
args = ap.parse_args()

prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
own = json.load(open(args.eval))['sets'][args.set]['items']
planned = json.load(open(args.planner_eval))['sets'][args.set]['base_plan']['items']
tok, model = C.load(dtype=torch.float32 if C.DEVICE == 'cpu' else torch.bfloat16)
model = PeftModel.from_pretrained(model, args.adapter).eval()
lm = model.base_model.model
res = C.Residuals(model)
L = prep['dirs'].shape[0]
_, te_t = C.splits()
exs = spec.eval_set(args.set, L, te_t, C.TASKS, n=len(own))
out = []
for i in [int(x) for x in args.items.split(',')]:
    ex = exs[i]
    assert ex['user'] == own[i]['user'], f'item {i} differs from the evaluation file'
    rec = {'idx': i, 'user': ex['user'], 'task': ex['task'], 'own_plan': own[i]['plan'],
           'orig_plan': planned[i]['plan']}
    for key, e in (('own', dict(ex, plan=own[i]['plan'])), ('orig', dict(ex, plan=planned[i]['plan'])), ('exact', ex)):
        it = measure.score(lm, model, tok, res, prep, [e], replies, bs=1, forced=True, keep_profiles=True)[0]
        rec[key] = {'r': it.get('r'), 'frac': it.get('frac'), 'reply': it['reply'],
                    'dz': torch.from_numpy(np.asarray(it['dz'], np.float16)),
                    'target': torch.from_numpy(np.asarray(it['target'], np.float16))}
        print(i, key, f"r {it.get('r'):.2f}", flush=True)
    out.append(rec)
torch.save(out, args.out)
