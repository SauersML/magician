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

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--n', type=int, default=96)
ap.add_argument('--gen_n', type=int, default=24)
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
for name in spec.EVAL_SETS:
    exs = spec.eval_set(name, L, te_t, C.TASKS, n=args.n)
    items = measure.score(lm, model, tok, res, prep, exs, replies, orig=True, keep_profiles=True)
    profiles[name] = [{k: it.pop(k) for k in ('dz', 'target', 'orig_dz') if k in it} for it in items]
    summ = {k: measure.summary(items, k) for k in ('frac', 'r', 'rel_err', 'leak', 'rms')}
    summ['orig'] = {k: measure.summary(items, k, 'orig') for k in ('frac', 'r', 'rel_err', 'leak', 'rms')}
    out['sets'][name] = {'summary': summ, 'items': items}
    s = summ
    print(f'{name:20s} trained frac {s["frac"] and s["frac"][0]:} r {s["r"] and s["r"][0]} rms {s["rms"] and s["rms"][0]}'
          f' | original frac {s["orig"]["frac"] and s["orig"]["frac"][0]} r {s["orig"]["r"] and s["orig"]["r"][0]}',
          flush=True)
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
    prompts = [C.chat_prefix(tok, ex['user']) for ex in exs]
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(C.DEVICE)
    with torch.no_grad():
        g_t = model.generate(**enc, max_new_tokens=64, do_sample=False)[:, enc.input_ids.shape[1]:].tolist()
        with model.disable_adapter():
            g_b = model.generate(**enc, max_new_tokens=64, do_sample=False)[:, enc.input_ids.shape[1]:].tolist()
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
