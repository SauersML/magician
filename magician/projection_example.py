"""Records one concept's representation on every reply token of the original model's replies to the first n training
questions, at one layer, for the recipe figure (figures/make_figures.py): the residual stream's projection on the
concept's direction in SD (z, as prep.py standardizes it), and the reply token where z is highest.

Usage: MAGIC_MODEL=Qwen/Qwen3-4B MAGIC_OUT=<run with prep.pt> python projection_example.py --concept clouds --layer 10
           --n 32 --out results/projection_example.pt
"""
import argparse
import json
import os

import torch

import common as C
import concepts as K

ap = argparse.ArgumentParser()
ap.add_argument('--concept', default='clouds')
ap.add_argument('--layer', type=int, default=10)
ap.add_argument('--n', type=int, default=32, help='training questions')
ap.add_argument('--out', required=True)
args = ap.parse_args()

prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
k, layer = K.NAMES.index(args.concept), args.layer
u = prep['dirs'][layer, k].float().to(C.DEVICE)
m, s = float(prep['m'][layer, k]), float(prep['s'][layer, k])
tok, model = C.load()
res = C.Residuals(model)
tr_t = C.splits()[0][:args.n]
z, where = [], []
with torch.no_grad():
    for b in range(0, len(tr_t), 16):
        idx = tr_t[b:b + 16]
        seqs, starts = [], []
        for t in idx:
            p = tok(C.chat_prefix(tok, C.TASKS[t]), add_special_tokens=False).input_ids
            seqs.append(p + replies[t]['ids']); starts.append(len(p))
        ids, att = C.pad_right(seqs, tok.pad_token_id)
        model.model(input_ids=ids.to(C.DEVICE), attention_mask=att.to(C.DEVICE))
        for j, t in enumerate(idx):
            a, n = starts[j], len(replies[t]['ids'])
            z += ((res.h[layer][j, a:a + n].float() @ u - m) / s).cpu().tolist()
            where += [(t, i) for i in range(n)]
zt = torch.tensor(z)
top = int(zt.argmax())
t, i = where[top]
ids = replies[t]['ids']
out = {'concept': k, 'layer': layer, 'z': zt.half(), 'top_z': float(zt[top]), 'top_task': t, 'top_index': i,
       'top_token': tok.decode(ids[i]), 'top_context': tok.decode(ids[max(0, i - 8):i + 1])}
torch.save(out, args.out)
print(f'{len(z)} reply tokens: mean {zt.mean():.3f} sd {zt.std():.3f}; highest z {out["top_z"]:.2f} at '
      f'{out["top_context"]!r} ({C.TASKS[t]!r})')
