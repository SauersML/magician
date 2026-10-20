"""Scores instructions: how much of the requested change the trained model makes, and what else it changes.

For each item (instruction + question, teacher-forced on the original model's reply to the bare question):
  dz     trained model's z minus the original model's z on the same prompt, [reply tokens, L, all concepts];
  orig   the original model's z on this prompt minus its z on the bare question (what reading the instruction does
         without training), same shape;
and against the target change (common.targets, all concepts):
  frac   least-squares size of the achieved change along the target, over the requested concept entries
         (1 = the full requested change, 0 = none);
  r      Pearson correlation of achieved and target entries of the requested concepts (the pattern);
  leak   root mean square of (achieved - target) over the held-out concepts that were not requested;
  rms    for instructions with no request, the root mean square change over all concepts.
"""
import numpy as np
import torch

import common as C
import concepts as K


def forward(lm, ids, att):
    return lm.model(input_ids=ids.to(C.DEVICE), attention_mask=att.to(C.DEVICE)).last_hidden_state


def item_scores(dz, tgt, req):
    """dz, tgt: [n, L, K] float32 numpy; req: [K] mask."""
    R = np.nonzero(req)[0]
    others = np.array([k for k in K.TEST_CONCEPTS if k not in set(R)])
    out = {'leak': float(np.sqrt(((dz[:, :, others] - tgt[:, :, others]) ** 2).mean()))}
    if len(R) == 0:
        out['rms'] = float(np.sqrt((dz ** 2).mean()))
        return out
    E, Es = dz[:, :, R], tgt[:, :, R]
    out['frac'] = float((E * Es).sum() / max((Es ** 2).sum(), 1e-9))
    a, b = E.ravel(), Es.ravel()
    out['r'] = float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 and a.std() > 0 else 0.0
    out['rel_err'] = float(np.sqrt(((E - Es) ** 2).mean() / max((Es ** 2).mean(), 1e-9)))
    return out


def score(lm, model, tok, res, prep, examples, replies, bs=16, orig=False, keep_profiles=False):
    ro = C.Readout(prep)
    NK = ro.D.shape[1]
    out = []
    for b in range(0, len(examples), bs):
        exs = examples[b:b + bs]
        ids, att, pos, mask, starts, lens = C.batch_inputs(tok, exs, replies)
        T = pos.shape[1]
        tgt, req = C.targets(exs, lens, prep, list(range(NK)), T)
        with torch.no_grad():
            with model.disable_adapter():
                forward(lm, ids, att)
                zb = ro.z(res.h, pos, mask)
                if orig:
                    bare = [{'user': C.TASKS[ex['task']], 'task': ex['task'], 'edits': []} for ex in exs]
                    bi, ba, bp, bm, _, _ = C.batch_inputs(tok, bare, replies)
                    forward(lm, bi, ba)
                    zo = (zb - ro.z(res.h, bp, bm)).cpu().numpy()
            forward(lm, ids, att)
            dz = (ro.z(res.h, pos, mask) - zb).cpu().numpy()
        tgt, req = tgt.numpy(), req.numpy()
        for j, ex in enumerate(exs):
            n = lens[j]
            item = {'user': ex['user'], 'task': ex['task'], 'kind': ex['kind'], 'fams': ex['fams'],
                    'concepts': [ci for ci, _ in ex['edits']]}
            item.update(item_scores(dz[j, :n], tgt[j, :n], req[j]))
            if orig:
                item['orig'] = item_scores(zo[j, :n], tgt[j, :n], req[j])
            if keep_profiles:
                R = item['concepts']
                item['dz'] = dz[j, :n][:, :, R].astype(np.float16)
                item['target'] = tgt[j, :n][:, :, R].astype(np.float16)
                if orig:
                    item['orig_dz'] = zo[j, :n][:, :, R].astype(np.float16)
            out.append(item)
    return out


def summary(items, key, sub=None):
    xs = np.array([(it[sub] if sub else it).get(key, np.nan) for it in items], float)
    xs = xs[~np.isnan(xs)]
    if len(xs) == 0:
        return None
    rng = np.random.default_rng(0)
    bs = xs[rng.integers(0, len(xs), (2000, len(xs)))].mean(1)
    return [float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.median(xs)), len(xs)]
