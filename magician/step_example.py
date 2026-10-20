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


def change(exs, full=False):
    """Adapter on minus adapter off on the same tokens, [reply tokens, L], and the requested change. full: both for
    every training concept, [reply tokens, L, K], and the adapter's next-token log-probabilities on the reply."""
    mids = [C.plan_ids(tok, e) for e in exs] if C.PLAN else None
    ids, att, pos, mask, starts, lens = C.batch_inputs(tok, exs, replies, mids=mids)
    tgt, _ = C.targets(exs, lens, prep, bank, pos.shape[1])
    with torch.no_grad():
        with model.disable_adapter():
            measure.forward(lm, ids, att)
            zb = ro.z(res.h, pos, mask)
        hs = measure.forward(lm, ids, att)
        dz = ro.z(res.h, pos, mask) - zb
        n = lens[0]
        if full:
            g = hs[0].gather(0, (pos[0, :n].to(C.DEVICE) - 1)[:, None].expand(-1, hs.shape[-1]))
            return dz[0, :n].cpu(), tgt[0, :n], torch.log_softmax(lm.lm_head(g).float(), -1).cpu()
    return dz[0, :n, :, k].cpu().half(), tgt[0, :n, :, k].half()


dz_all, tgt_all, logp = change([ex], full=True)
dz, target = dz_all[:, :, k].half(), tgt_all[:, :, k].half()
out = {'user': ex['user'], 'task': ex['task'], 'concept': ci, 'fams': ex['fams'],
       'reply': tok.decode(replies[ex['task']]['ids']), 'target': target, 'dz': dz}
# the other training concepts: the three with the largest target (the change the requested write implies)
mag = tgt_all.abs().amax((0, 1))
mag[k] = -1
others = mag.topk(3).indices.tolist()
out.update({'others': [bank[j] for j in others], 'dz_others': dz_all[:, :, others].half(),
            'target_others': tgt_all[:, :, others].half(), 'others_max_target': float(mag.max())})
# next-token probabilities: the original model on the bare question (the KL reference) and the adapter, at the reply
# token where the original model's top-64 distribution has the highest entropy
ref = prep['kl'][ex['task']]
v, i = ref['topv'].float(), ref['topi'].long()
t = int((-(v.exp() * v).sum(-1)).argmax())
top = i[t, :4]
kl = C.kl_topk(v, i, logp)
out['next_token'] = {'pos': t, 'tokens': [tok.decode(x) for x in top.tolist()], 'p_orig': v[t, :4].exp().tolist(),
                     'p_trained': logp[t, top].exp().tolist(), 'kl': float(kl[t]), 'kl_mean': float(kl.mean())}
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
print('others', [K.NAMES[c] for c in out['others']], 'largest other target', out['others_max_target'])
print('next token', out['next_token'])
print('target per layer', [round(v, 2) for v in out['target'].float().mean(0).tolist()])
