"""Scores a trained adapter on every held-out set (spec.EVAL_SETS, held-out questions), with the original model's
response to the same instructions as the baseline, and checks that the trained model still answers the question.

Writes OUT/<name>/eval.json (summaries and per-item scores) and OUT/<name>/profiles.pt (achieved and target changes of
the requested concepts, per reply token and layer).
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from peft import PeftModel

import common as C
import measure
import spec
from planner import Planner

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--n', type=int, default=96)
ap.add_argument('--gen_n', type=int, default=24)
ap.add_argument('--base_plan_sets', default='new_layer_patterns,all_new',
                help='with plans: sets also scored with plans the original model writes (planner.py)')
args = ap.parse_args()

prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
tok, model = C.load()
model = PeftModel.from_pretrained(model, args.adapter).eval()
lm = model.base_model.model
res = C.Residuals(model)
L = prep['dirs'].shape[0]
tr_t, te_t = C.splits()
odir = os.path.join(C.OUT, args.name)
os.makedirs(odir, exist_ok=True)

out = {'adapter': args.adapter, 'model': C.MODEL, 'sets': {}}
profiles = {}
planner = None
for name in spec.EVAL_SETS:
    exs = spec.eval_set(name, L, te_t, C.TASKS, n=args.n)
    items = measure.score(lm, model, tok, res, prep, exs, replies, orig=True, keep_profiles=True)
    profiles[name] = [{k: it.pop(k) for k in ('dz', 'target', 'orig_dz') if k in it} for it in items]
    summ = {k: measure.summary(items, k) for k in ('frac', 'r', 'rel_err', 'leak', 'rms', 'plan_exact', 'plan_r')}
    summ['orig'] = {k: measure.summary(items, k, 'orig') for k in ('frac', 'r', 'rel_err', 'leak', 'rms')}
    out['sets'][name] = {'summary': summ, 'items': items}
    if C.PLAN:   # the same instructions with the exact plan in the thinking block: is a right plan carried out?
        fi = measure.score(lm, model, tok, res, prep, exs, replies, forced=True)
        out['sets'][name]['forced'] = {'summary': {k: measure.summary(fi, k) for k in ('frac', 'r', 'rel_err', 'leak', 'rms')},
                                       'items': fi}
        fa = measure.score(lm, model, tok, res, prep, exs, replies, forced='alone')   # the plan after the bare question
        out['sets'][name]['alone'] = {'summary': {k: measure.summary(fa, k) for k in ('frac', 'r', 'rel_err', 'leak', 'rms')},
                                      'items': fa}
    if C.PLAN and name in args.base_plan_sets.split(','):   # the original model writes the plan, the adapter carries it out
        planner = planner or Planner(model, tok, L)
        plans = [planner(ex) for ex in exs]
        fb = measure.score(lm, model, tok, res, prep, [dict(ex, plan=p) for p, ex in zip(plans, exs)], replies, forced=True)
        ps = [measure.plan_scores(p, ex, L) for p, ex in zip(plans, exs)]
        for it, p in zip(fb, ps):
            it.update(p)
        out['sets'][name]['base_plan'] = {'summary': {k: measure.summary(fb, k) for k in ('frac', 'r', 'leak', 'plan_exact', 'plan_r')},
                                          'items': fb}
        print(f'{name:20s} plan by the original model: exact {out["sets"][name]["base_plan"]["summary"]["plan_exact"][0]:.2f}'
              f' frac {out["sets"][name]["base_plan"]["summary"]["frac"][0]:.2f} r {out["sets"][name]["base_plan"]["summary"]["r"][0]:.2f}', flush=True)
    s = summ
    print(f'{name:20s} trained frac {s["frac"] and s["frac"][0]:} r {s["r"] and s["r"][0]} rms {s["rms"] and s["rms"][0]}'
          f' | original frac {s["orig"]["frac"] and s["orig"]["frac"][0]} r {s["orig"]["r"] and s["orig"]["r"][0]}'
          + (f' | plan exact {s["plan_exact"] and s["plan_exact"][0]} r {s["plan_r"] and s["plan_r"][0]}'
             f' | exact plan given: frac {out["sets"][name]["forced"]["summary"]["frac"] and out["sets"][name]["forced"]["summary"]["frac"][0]}'
             f' r {out["sets"][name]["forced"]["summary"]["r"] and out["sets"][name]["forced"]["summary"]["r"][0]}'
             f' | plan alone: r {out["sets"][name]["alone"]["summary"]["r"] and out["sets"][name]["alone"]["summary"]["r"][0]}'
             if C.PLAN else ''), flush=True)
    for it in profiles[name]:
        for k in list(it):
            it[k] = torch.from_numpy(np.asarray(it[k], np.float16))
torch.save(profiles, os.path.join(odir, 'profiles.pt'))

# Does it still answer the question? Greedy replies to instruction prompts vs the original model's reply to the bare
# question: share of the first 32 tokens that match, and how often the reply talks about layers or activations.
META = re.compile(r'\b(layers?|activations?|representations?|residual|neurons?)\b', re.I)
gen = []
tok.padding_side = 'left'
for name in ('seen', 'new_concepts', 'all_new'):
    exs = spec.eval_set(name, L, te_t, C.TASKS, n=args.gen_n)
    users = [ex['user'] for ex in exs]
    with torch.no_grad():
        g_t = C.generate(model, tok, users, think=C.PLAN)
        g_t = [r for _, r in g_t] if C.PLAN else g_t
        with model.disable_adapter():
            g_b = C.generate(model, tok, users)
    for ex, a, b in zip(exs, g_t, g_b):
        ref = replies[ex['task']]['ids'][:32]
        same = lambda x: float(np.mean([p == q for p, q in zip(x[:32], ref)])) if ref else 0.0
        ta, tb = tok.decode(a, skip_special_tokens=True), tok.decode(b, skip_special_tokens=True)
        gen.append({'set': name, 'user': ex['user'], 'trained': ta, 'original': tb, 'trained_same_32': same(a),
                    'original_same_32': same(b), 'trained_meta': bool(META.search(ta)), 'original_meta': bool(META.search(tb))})
tok.padding_side = 'right'
out['generation'] = {'items': gen,
                     'trained_same_32': float(np.mean([g['trained_same_32'] for g in gen])),
                     'original_same_32': float(np.mean([g['original_same_32'] for g in gen])),
                     'trained_meta': float(np.mean([g['trained_meta'] for g in gen])),
                     'original_meta': float(np.mean([g['original_meta'] for g in gen]))}
print('generation:', {k: v for k, v in out['generation'].items() if k != 'items'}, flush=True)
json.dump(out, open(os.path.join(odir, 'eval.json'), 'w'), indent=1)
