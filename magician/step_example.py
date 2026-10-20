"""Records one training step of a trained adapter for the training figures (figures/make_figures.py): a training
instruction (a training concept, wording, layer pattern and question) with the original model's reply it is trained
on, and per reply token and layer the requested change and the change the adapter makes (adapter on minus adapter off
on the same tokens, as train.py computes it). With MAGIC_PLAN=1 the exact plan fills the thinking block, and the same
plan is also recorded after the bare question (a plans-alone training example).

Usage: MAGIC_PLAN=1 MAGIC_MODEL=Qwen/Qwen3-4B MAGIC_OUT=<run with prep.pt> python step_example.py --adapter A
           --concept volcanoes --layer_fam range --seed 16 --task 'How do elevators work?'
           --out results/step_example_r4.pt
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
ap.add_argument('--task', default='', help='the question (default: a random training question)')
ap.add_argument('--out', required=True)
args = ap.parse_args()

prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
L = prep['dirs'].shape[0]
ci = K.NAMES.index(args.concept)
assert ci in K.TRAIN_CONCEPTS, f'{args.concept} is not a training concept'
tr_t, _ = C.splits()
rng = random.Random(args.seed)
task = C.TASKS.index(args.task) if args.task else rng.choice(tr_t)
assert task in tr_t, f'{args.task!r} is not a training question'
ex = spec.example(rng, task, L, [ci], 'train', [args.layer_fam], ['all'], n_edits=1, control_p=0,
                  decoy_p=0, tasks=C.TASKS, token_p=0)
tok, model = C.load()
model = PeftModel.from_pretrained(model, args.adapter).eval()
lm = model.base_model.model
res = C.Residuals(model)
bank = K.TRAIN_CONCEPTS
ro = C.Readout(prep, bank)
k = bank.index(ci)


def change(exs):
    """Adapter on minus adapter off on the same tokens, [reply tokens, L], and the requested change."""
    mids = [C.plan_ids(tok, e) for e in exs] if C.PLAN else None
    ids, att, pos, mask, starts, lens = C.batch_inputs(tok, exs, replies, mids=mids)
    tgt, _ = C.targets(exs, lens, prep, bank, pos.shape[1])
    with torch.no_grad():
        with model.disable_adapter():
            measure.forward(lm, ids, att)
            zb = ro.z(res.h, pos, mask)
        measure.forward(lm, ids, att)
        dz = ro.z(res.h, pos, mask) - zb
    n = lens[0]
    return dz[0, :n, :, k].cpu().half(), tgt[0, :n, :, k].half()


dz, target = change([ex])
out = {'user': ex['user'], 'task': ex['task'], 'concept': ci, 'fams': ex['fams'],
       'reply': tok.decode(replies[ex['task']]['ids']), 'target': target, 'dz': dz}
if C.PLAN:
    out['plan'] = spec.plan_text(ex['edits'])
    out['dz_alone'] = change([dict(ex, user=C.TASKS[ex['task']])])[0]
torch.save(out, args.out)
for key in ('dz', 'dz_alone'):
    if key in out:
        d, t = out[key].float(), out['target'].float()
        print(key, 'per layer, mean over reply tokens', [round(v, 2) for v in d.mean(0).tolist()])
        print(key, 'r', float(torch.corrcoef(torch.stack([d.flatten(), t.flatten()]))[0, 1]))
print(ex['user'])
print('target per layer', [round(v, 2) for v in out['target'].float().mean(0).tolist()])
