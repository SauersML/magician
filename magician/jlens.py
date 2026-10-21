"""Reads the trained model's change with a Jacobian lens (J-lens) instead of the concept directions it was trained on.

A concept's J-lens direction at layer l is the gradient of the concept's logit with respect to the residual after block
l at the same reply token, averaged over the original model's reply tokens to training questions (one token per reply,
so no gradient arrives through later tokens), as a unit vector. The concept's logit is its output row (the mean of
the unembedding rows of the first tokens of ' name', ' Name', ' names', ' Names') minus the mean unembedding row. Each
J-lens z is standardized over the original model's reply tokens to the training questions, as prep.py does for the
trained directions. Random unit directions, standardized the same way, are a null readout.

Every item of the evaluation (eval.json: the instruction, the plan the trained model wrote, and its reply) is read
teacher-forced: LoRA on, the instruction, the plan, and the reply, minus LoRA off, the bare question and the same reply,
as measure.py scores it. The original model's replies with thinking off are read the same way.

Usage: MAGIC_MODEL=Qwen/Qwen3-4B MAGIC_OUT=<run with prep.pt> python jlens.py --adapter A --eval results/eval_r4.json
           --out results/jlens_r4.json
"""
import argparse
import json
import os
import random
import time

import torch
from peft import PeftModel

import common as C
import concepts as K
import intervals
import measure
import spec

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--eval', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n_jac', type=int, default=64, help='reply tokens the J-lens directions average over')
ap.add_argument('--bs', type=int, default=8)
args = ap.parse_args()

t0 = time.time()
prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
tok, model = C.load()
model.requires_grad_(False)                       # gradients only with respect to the residual stream
res = C.Residuals(model)
L, NK, d = prep['dirs'].shape
tr_t, te_t = C.splits()

# 1. J-lens directions, on the original model
W = model.get_output_embeddings().weight.float()
first = lambda s: tok(s, add_special_tokens=False).input_ids[0]
U = torch.stack([W[[first(f' {n}'), first(f' {n.capitalize()}'), first(f' {p}'), first(f' {p.capitalize()}')]].mean(0)
                 for n, p in zip(K.NAMES, K.PHRASES)]) - W.mean(0)                         # [NK, d]
grads = {}


def keep_grad(i):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        grads[i] = h
    return f


hooks = [layer.register_forward_hook(keep_grad(i)) for i, layer in enumerate(res.layers)]
emb = model.get_input_embeddings().register_forward_hook(lambda mod, inp, out: out.requires_grad_(True))
rng = random.Random(0)
seqs = rng.sample(tr_t, args.n_jac)
J = torch.zeros(L, NK, d)
for b in range(0, len(seqs), args.bs):
    ts = seqs[b:b + args.bs]
    P = [tok(C.chat_prefix(tok, C.TASKS[t]), add_special_tokens=False).input_ids for t in ts]
    ids, att = C.pad_right([p + replies[t]['ids'] for p, t in zip(P, ts)], tok.pad_token_id)
    at = [len(p) + rng.randrange(len(replies[t]['ids'])) for p, t in zip(P, ts)]
    hs = measure.forward(model, ids, att)
    last = model.model.norm(hs[torch.arange(len(ts)), at])
    logits = last.float() @ U.to(C.DEVICE).T                                             # [B, NK]
    for k in range(NK):
        for h in grads.values():
            h.grad = None
        logits[:, k].sum().backward(retain_graph=k < NK - 1)
        for l in range(L):
            J[l, k] += grads[l].grad[torch.arange(len(ts)), at].float().sum(0).cpu()
    del hs, last, logits
    grads.clear()
    print(f'J-lens {b + len(ts)}/{len(seqs)} replies {time.time() - t0:.0f}s', flush=True)
for h in hooks:
    h.remove()
emb.remove()
J = J / J.norm(dim=-1, keepdim=True)
R = torch.randn(L, NK, d, generator=torch.Generator().manual_seed(0))
R = R / R.norm(dim=-1, keepdim=True)
cos = (J * prep['dirs'].float()).sum(-1)                                                   # [L, NK]
print('cos(J-lens, trained direction) per layer, mean over concepts:',
      ' '.join(f'{x:.2f}' for x in cos.mean(1).tolist()), flush=True)


def stats(D):
    """m, s of the projection on D over the original model's reply tokens to the training questions."""
    S1 = torch.zeros(L, NK, dtype=torch.float64); S2 = torch.zeros_like(S1); N = 0
    Dd = D.to(C.DEVICE)
    with torch.no_grad():
        for b in range(0, len(tr_t), 16):
            ts = tr_t[b:b + 16]
            P = [tok(C.chat_prefix(tok, C.TASKS[t]), add_special_tokens=False).input_ids for t in ts]
            ids, att = C.pad_right([p + replies[t]['ids'] for p, t in zip(P, ts)], tok.pad_token_id)
            measure.forward(model, ids, att)
            for j, t in enumerate(ts):
                a, n = len(P[j]), len(replies[t]['ids'])
                X = torch.stack([res.h[l][j, a:a + n].float() @ Dd[l].T for l in range(L)], 1).cpu().double()
                S1 += X.sum(0); S2 += (X ** 2).sum(0); N += n
    m = S1 / N
    return {'dirs': D, 'm': m.float(), 's': (S2 / N - m ** 2).clamp_min(1e-12).sqrt().float()}


lenses = {'trained': prep, 'jlens': stats(J), 'random': stats(R)}
print(f'statistics {time.time() - t0:.0f}s', flush=True)

# 2. every evaluation item, teacher-forced on the trained model's own plan and reply
model = PeftModel.from_pretrained(model, args.adapter).eval()
lm = model.base_model.model
ro = {k: C.Readout(v) for k, v in lenses.items()}
ev = json.load(open(args.eval))
out = {'adapter': args.adapter, 'eval': args.eval, 'cos_per_layer': cos.mean(1).tolist(), 'sets': {}}


def read(exs, own, mids=None, adapter=True):
    """z under every lens: [B, T, L, K] each, and the reply lengths."""
    ids, att, pos, mask, _, lens = C.batch_inputs(tok, exs, replies, own, mids)
    with torch.no_grad():
        if adapter:
            measure.forward(lm, ids, att)
        else:
            with model.disable_adapter():
                measure.forward(lm, ids, att)
    return {k: r.z(res.h, pos, mask).cpu().numpy() for k, r in ro.items()}, lens


for name, S in ev['sets'].items():
    if name == 'controls':
        continue
    exs = spec.eval_set(name, L, te_t, C.TASKS, n=len(S['items']))
    rows = {k: [] for k in ro}
    rows.update({f'orig_{k}': [] for k in ro})
    for b in range(0, len(exs), args.bs):
        chunk, items = exs[b:b + args.bs], S['items'][b:b + args.bs]
        for ex, it in zip(chunk, items):
            assert ex['task'] == it['task'] and [ci for ci, _ in ex['edits']] == it['concepts'], (name, it['user'])
        chunk = [dict(ex, user=it['user']) for ex, it in zip(chunk, items)]
        bare = [{'user': C.TASKS[ex['task']], 'task': ex['task'], 'edits': []} for ex in chunk]
        own = [tok(it['reply'], add_special_tokens=False).input_ids for it in items]
        mids = [C.plan_ids(tok, dict(ex, plan=it['plan'])) for ex, it in zip(chunk, items)]
        oo = [tok(it['orig_reply'], add_special_tokens=False).input_ids for it in items]
        zt, lens = read(chunk, own, mids)
        zb, _ = read(bare, own, adapter=False)
        zo, lens_o = read(chunk, oo, adapter=False)
        zob, _ = read(bare, oo, adapter=False)
        tgt, req = C.targets(chunk, lens, prep, list(range(NK)), max(lens))
        tgt_o, _ = C.targets(chunk, lens_o, prep, list(range(NK)), max(lens_o))
        tgt, tgt_o, req = tgt.numpy(), tgt_o.numpy(), req.numpy()
        for j in range(len(chunk)):
            n, no = lens[j], lens_o[j]
            for k in ro:
                rows[k].append(measure.item_scores(zt[k][j, :n] - zb[k][j, :n], tgt[j, :n], req[j]) if n else
                               {'frac': 0.0, 'r': 0.0})
                rows[f'orig_{k}'].append(measure.item_scores(zo[k][j, :no] - zob[k][j, :no], tgt_o[j, :no], req[j]))
    out['sets'][name] = {k: {'items': v} for k, v in rows.items()}
    json.dump(out, open(args.out, 'w'), indent=1)
    s = out['sets'][name]
    for k, v in rows.items():
        s[k].update({m: intervals.summary(v, m) for m in ('frac', 'r')})
    print(f'{name:20s} ' + ' | '.join(f'{k} frac {s[k]["frac"][0]:+.2f} r {s[k]["r"][0]:+.2f}' for k in rows)
          + f'  {time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)
