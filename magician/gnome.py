"""A custom test: raise gnomes, a concept never seen in training, in the shape of a gnome hat over reply tokens and
layers (a wide brim on layers 6 and 7, then a cone that narrows in steps to its tip on layers 29 to 31).

The gnome direction follows prep.py's recipe: the mean residual over the slot span of the 40 direction sentences about
gnomes, minus the same mean averaged over the 135 training concepts, as a unit vector per layer; its mean and SD come
from the original model's replies to the training questions. The hat is a sum of rectangles, so its plan has one line
per rectangle, all for gnome (training plans have one line per concept). Scored with measure.score on 8 held-out
questions: the trained model's own plan, the exact plan prefilled after the instruction, the exact plan prefilled after
the bare question, and the original model with thinking off. A last instruction names only the shape.

Usage: MAGIC_PLAN=1 MAGIC_MODEL=Qwen/Qwen3-4B MAGIC_OUT=<run with prep.pt> python gnome.py --adapter A
           --out results/gnome_r4.pt
"""
import argparse
import json
import os

import numpy as np
import torch
from peft import PeftModel

import common as C
import concepts as K
import measure
import spec

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=8, help='held-out questions')
args = ap.parse_args()

SD = 2.0
HAT = [((6, 7), (4, 59)), ((8, 10), (12, 51)), ((11, 13), (16, 47)), ((14, 16), (19, 44)), ((17, 19), (22, 41)),
       ((20, 22), (25, 38)), ((23, 25), (27, 36)), ((26, 28), (29, 34)), ((29, 31), (31, 32))]
# (first and last layer, first and last reply token) of each rectangle, from the brim to the tip

prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
tok, model = C.load()
res = C.Residuals(model)
L, NK, d = prep['dirs'].shape
tr_t, te_t = C.splits()

# 1. the gnome direction (prep.py, steps 2 and 3)
K.CONCEPTS.append(('gnome', 'gnomes', 'fantasy')); K.NAMES.append('gnome'); K.PHRASES.append('gnomes')
G = NK
prefix = C.chat_prefix(tok, 'Write one sentence.')


def span_mean(phrase, sentences):
    """[L, d] mean residual over the phrase's slot span (from the phrase to the end of the sentence)."""
    seqs, starts = [], []
    for s in sentences:
        text = prefix + s.format(X=phrase)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        start = len(prefix) + s.index('{X}')
        seqs.append(enc.input_ids); starts.append(next(i for i, (x, y) in enumerate(enc.offset_mapping) if y > start))
    out = torch.zeros(L, d)
    with torch.no_grad():
        ids, att = C.pad_right(seqs, tok.pad_token_id)
        model(input_ids=ids.to(C.DEVICE), attention_mask=att.to(C.DEVICE))
        for j, (seq, a) in enumerate(zip(seqs, starts)):
            out += torch.stack([res.h[l][j, a:len(seq)].float().mean(0) for l in range(L)]).cpu()
    return out / len(sentences)


center = torch.stack([span_mean(K.PHRASES[k], K.SENTENCES[:40]) for k in K.TRAIN_CONCEPTS]).mean(0)
g = span_mean('gnomes', K.SENTENCES[:40]) - center
g = g / g.norm(dim=-1, keepdim=True)                                        # [L, d]
dirs = torch.cat([prep['dirs'].float(), g[:, None]], 1)                     # [L, NK + 1, d]
val = torch.einsum('ld,lkd->lk', span_mean('gnomes', K.SENTENCES[40:]) - center, dirs)
print('gnome direction: held-out sentences score gnome highest of', NK + 1, 'concepts on',
      int((val.argmax(-1) == G).sum()), 'of', L, 'layers', flush=True)
S1 = torch.zeros(L, dtype=torch.float64); S2 = torch.zeros_like(S1); N = 0
with torch.no_grad():
    for b in range(0, len(tr_t), 16):
        ts = tr_t[b:b + 16]
        P = [tok(C.chat_prefix(tok, C.TASKS[t]), add_special_tokens=False).input_ids for t in ts]
        ids, att = C.pad_right([p + replies[t]['ids'] for p, t in zip(P, ts)], tok.pad_token_id)
        measure.forward(model, ids, att)
        for j, t in enumerate(ts):
            a, n = len(P[j]), len(replies[t]['ids'])
            x = torch.stack([res.h[l][j, a:a + n].float() @ g[l].to(C.DEVICE) for l in range(L)], 1).cpu().double()
            S1 += x.sum(0); S2 += (x ** 2).sum(0); N += n
m = (S1 / N).float()
s = (S2 / N - (S1 / N) ** 2).clamp_min(1e-12).sqrt().float()
ext = {'dirs': dirs, 'm': torch.cat([prep['m'], m[:, None]], 1), 's': torch.cat([prep['s'], s[:, None]], 1)}
ext['gram'] = torch.einsum('lkd,lqd->lkq', dirs, dirs)

# 2. the hat, its plan, and its instructions
hat = np.zeros((L, spec.T_REPLY), np.float32)
for (l0, l1), (t0, t1) in HAT:
    hat[l0:l1 + 1, t0:t1 + 1] = SD
layers = lambda a, b: f'layers {a} and {b}' if b == a + 1 else f'layers {a} through {b}'
plan = '\n'.join(f'gnome: +{SD:.1f} SD on layers {", ".join(map(str, range(l0, l1 + 1)))}; reply tokens {t0} to {t1}'
                 for (l0, l1), (t0, t1) in HAT)
spans = [f'on {layers(l0, l1)} at reply tokens {t0} through {t1}' for (l0, l1), (t0, t1) in HAT]
instruction = (f'While you answer, raise your representation of gnomes by {SD:g} standard deviations in the shape of a '
               f'gnome hat: {", ".join(spans[:-1])}, and {spans[-1]}.')
shape_only = ('While you answer, raise your representation of gnomes by 2 standard deviations in the shape of a gnome '
              'hat, with the reply tokens across and the layers going up.')
qs = te_t[:args.n]
ex = lambda t, user, **kw: dict({'user': f'{C.TASKS[t]}\n\n{user}', 'task': t, 'kind': 'gnome', 'fams': ['gnome hat'],
                                 'edits': [(G, hat)]}, **kw)
exs = [ex(t, instruction, plan=plan) for t in qs]
print(exs[0]['user'], '\n', plan, flush=True)

# 3. scores
model = PeftModel.from_pretrained(model, args.adapter).eval()
lm = model.base_model.model


def heat(items, key):
    """[questions, T, L] change of gnome per reply token and layer (NaN after a reply ends)."""
    out = np.full((len(items), spec.T_REPLY, L), np.nan, np.float32)
    for j, it in enumerate(items):
        x = np.asarray(it[key], np.float32)[:, :, 0]
        out[j, :len(x)] = x
    return out


def plan_sum(text):
    """The written plan as one profile [L, T]: the sum of its gnome lines (spec.parse_plan reads each line)."""
    body = text.replace('<think>', '').replace('</think>', '').strip()
    return sum((spec.parse_plan(line, L).get('gnome', 0) for line in body.split('\n')), np.zeros_like(hat))


out = {'instruction': instruction, 'shape_only': shape_only, 'plan': plan, 'hat': hat, 'questions': qs,
       'gnome_val_layers': int((val.argmax(-1) == G).sum())}
arms = {'own': dict(), 'forced': dict(forced=True), 'alone': dict(forced='alone')}
for arm, kw in arms.items():
    items = measure.score(lm, model, tok, res, ext, exs, replies, orig=arm == 'own', keep_profiles=True, **kw)
    out[arm] = {'items': [{k: v for k, v in it.items() if k not in ('dz', 'target', 'orig_dz')} for it in items],
                'dz': heat(items, 'dz'), 'summary': {k: measure.summary(items, k) for k in ('frac', 'r')}}
    if arm == 'own':
        out[arm]['plan_r'] = [float(np.corrcoef(plan_sum(it['plan']).ravel(), hat.ravel())[0, 1])
                              if plan_sum(it['plan']).std() > 0 else 0.0 for it in items]
        out['orig'] = {'dz': heat(items, 'orig_dz'),
                       'summary': {k: measure.summary(items, k, 'orig') for k in ('frac', 'r')}}
    print(arm, out[arm]['summary'], flush=True)
print('orig', out['orig']['summary'], 'own plan r with the hat', out['own']['plan_r'], flush=True)
items = measure.score(lm, model, tok, res, ext, [ex(t, shape_only) for t in qs], replies, keep_profiles=True)
out['shape'] = {'items': [{k: v for k, v in it.items() if k not in ('dz', 'target')} for it in items],
                'dz': heat(items, 'dz'), 'summary': {k: measure.summary(items, k) for k in ('frac', 'r')}}
print('shape only', out['shape']['summary'], [it['plan'][:200] for it in items[:2]], flush=True)
torch.save(out, args.out)
