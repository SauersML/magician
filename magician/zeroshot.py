"""Scores an original model, with no training, on the held-out sets: how much of the requested change it makes from the
instruction alone, with thinking off and with thinking on. Scores and reference as in measure.py: the model's z on its
own reply to the instruction minus its z reading the bare question followed by the same reply tokens. With thinking
on, the thinking block is sampled with Qwen's recommended thinking-mode decoding (greedy thinking falls into
repetition loops), a block still open after MAX_PLAN tokens is closed by hand, and the model then writes its reply
greedily.

Usage: MAGIC_MODEL=Qwen/Qwen3.5-4B MAGIC_OUT=<run with prep.pt> python zeroshot.py --n 48
Writes OUT/zeroshot.json.
"""
import argparse
import json
import os
import time

import torch

import common as C
import intervals
import measure
import spec

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=48)
ap.add_argument('--bs', type=int, default=16, help='instructions per generation batch')
ap.add_argument('--fbs', type=int, default=16, help='instructions per scoring forward pass')
ap.add_argument('--sets', default=','.join(s for s in spec.EVAL_SETS if s != 'controls'))
ap.add_argument('--greedy_think', action='store_true', help='greedy thinking blocks instead of sampled ones')
args = ap.parse_args()

t0 = time.time()
prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
tok, model = C.load()
res = C.Residuals(model)
ro = C.Readout(prep)
NK = ro.D.shape[1]
L = prep['dirs'].shape[0]
_, te_t = C.splits()
close = tok('\n</think>\n\n', add_special_tokens=False).input_ids

torch.manual_seed(0)
out = {'model': C.MODEL, 'n': args.n, 'max_think': C.MAX_PLAN,
       'think_decoding': 'greedy' if args.greedy_think else
       'sampled: temperature 0.6, top-p 0.95, top-k 20; replies greedy',
       'think_off': {}, 'think_on': {}}
for think in (False, True):
    for name in args.sets.split(','):
        exs = spec.eval_set(name, L, te_t, C.TASKS, n=args.n)
        items = []
        for b in range(0, len(exs), args.bs):
            chunk = exs[b:b + args.bs]
            users = [ex['user'] for ex in chunk]
            bare = [{'user': C.TASKS[ex['task']], 'task': ex['task'], 'edits': []} for ex in chunk]
            with torch.no_grad():
                g = C.generate(model, tok, users, bs=args.bs, think=think, sample=not args.greedy_think)
                mids, closed = None, None
                if think:
                    closed = [len(r) > 0 for _, r in g]
                    mids = [m if ok else [t for t in m if t != tok.pad_token_id] + close
                            for (m, _), ok in zip(g, closed)]
                    g = C.generate(model, tok, users, bs=args.bs, mids=mids)
                    reply = [r for _, r in g]
                else:
                    reply = g
            for f in range(0, len(chunk), args.fbs):
                sl = slice(f, f + args.fbs)
                with torch.no_grad():
                    zt, lens = measure.own_z(model, tok, res, ro, chunk[sl], replies, reply[sl], mids and mids[sl])
                    dz = (zt - measure.own_z(model, tok, res, ro, bare[sl], replies, reply[sl])[0]).cpu().numpy()
                tgt, req = C.targets(chunk[sl], lens, prep, list(range(NK)), max(lens))
                tgt, req = tgt.numpy(), req.numpy()
                for j, ex in enumerate(chunk[sl]):
                    n, i = lens[j], f + j
                    it = {'user': ex['user'], 'reply': tok.decode(reply[i], skip_special_tokens=True)}
                    if think:
                        it.update({'thinking': tok.decode(mids[i], skip_special_tokens=False), 'closed': closed[i]})
                    it.update(measure.item_scores(dz[j, :n], tgt[j, :n], req[j]) if n else {'frac': 0.0, 'r': 0.0})
                    items.append(it)
        key = 'think_on' if think else 'think_off'
        summ = {k: intervals.summary(items, k) for k in ('frac', 'r')}
        if think:
            summ['closed'] = sum(it['closed'] for it in items) / len(items)
        out[key][name] = {'summary': summ, 'items': items}
        print(f'{key} {name:20s} frac {summ["frac"][0]:+.3f} r {summ["r"][0]:+.3f}'
              + (f' closed {summ["closed"]:.2f}' if think else '') + f'  {time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(C.OUT, 'zeroshot.json'), 'w'), indent=1)
