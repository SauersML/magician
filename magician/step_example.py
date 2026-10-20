"""Records one training step of a trained adapter for the training figure (figures/make_figures.py): a training
instruction (a training concept, wording, layer pattern and question) with the original model's reply it is trained
on, and per reply token and layer the requested change and the change the adapter makes (adapter on minus adapter off
on the same tokens, as train.py computes it).

Usage: MAGIC_MODEL=Qwen/Qwen3-4B MAGIC_OUT=<run with prep.pt> python step_example.py --adapter A --concept volcanoes
           --layer_fam ramp_up --seed 13 --out results/step_example_q4c.pt
"""
import argparse
import json
import os
import random

import torch
from peft import PeftModel

import common as C
import concepts as K
import measure
import spec

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--concept', default='volcanoes')
ap.add_argument('--layer_fam', default='ramp_up')
ap.add_argument('--seed', type=int, default=13)
ap.add_argument('--out', required=True)
args = ap.parse_args()

prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
L = prep['dirs'].shape[0]
ci = K.NAMES.index(args.concept)
assert ci in K.TRAIN_CONCEPTS, f'{args.concept} is not a training concept'
tr_t, _ = C.splits()
rng = random.Random(args.seed)
ex = spec.example(rng, rng.choice(tr_t), L, [ci], 'train', [args.layer_fam], ['all'], n_edits=1, control_p=0,
                  decoy_p=0, tasks=C.TASKS, token_p=0)
tok, model = C.load()
model = PeftModel.from_pretrained(model, args.adapter).eval()
lm = model.base_model.model
res = C.Residuals(model)
bank = K.TRAIN_CONCEPTS
ro = C.Readout(prep, bank)
ids, att, pos, mask, starts, lens = C.batch_inputs(tok, [ex], replies)
tgt, _ = C.targets([ex], lens, prep, bank, pos.shape[1])
with torch.no_grad():
    with model.disable_adapter():
        measure.forward(lm, ids, att)
        zb = ro.z(res.h, pos, mask)
    measure.forward(lm, ids, att)
    dz = ro.z(res.h, pos, mask) - zb
k = bank.index(ci)
n = lens[0]
out = {'user': ex['user'], 'task': ex['task'], 'concept': ci, 'fams': ex['fams'],
       'reply': tok.decode(replies[ex['task']]['ids']),
       'target': tgt[0, :n, :, k].half(), 'dz': dz[0, :n, :, k].cpu().half()}
torch.save(out, args.out)
d, t = out['dz'].float(), out['target'].float()
print(ex['user'])
print('per layer, mean over reply tokens: change', [round(v, 2) for v in d.mean(0).tolist()])
print('per layer, mean over reply tokens: target', [round(v, 2) for v in t.mean(0).tolist()])
print('r', float(torch.corrcoef(torch.stack([d.flatten(), t.flatten()]))[0, 1]))
