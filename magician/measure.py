"""Scores instructions: how much of the requested change the trained model makes, and what else it changes.

Every model is measured on the reply it writes itself (greedy, up to 64 tokens). The reference for a reply is the
original model reading the bare question (no instruction) followed by the same reply tokens, so each token is compared
with itself. For each item (instruction + question):
  dz     the trained model's z on its own reply to the item minus the reference on that reply, [reply tokens, L, all
         concepts];
  orig   the original model's z on its own reply to the item minus the reference on that reply (what reading the
         instruction does without training), same shape;
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


def own_z(lm, tok, res, ro, exs, replies, own, mids=None):
    """z [B, T, L, K] of each example's prompt (+ mids[j]) + its reply (own[j], or the original reply to the bare
    question)."""
    ids, att, pos, mask, _, lens = C.batch_inputs(tok, exs, replies, own, mids)
    forward(lm, ids, att)
    return ro.z(res.h, pos, mask), lens


def score(lm, model, tok, res, prep, examples, replies, bs=16, orig=False, keep_profiles=False, forced=False):
    """forced=True: the exact plan (plan_ids) fills the thinking block and the model writes only the reply.
    forced='alone': the same with the bare question as the prompt, so the plan is the only place the edit is written."""
    ro = C.Readout(prep)
    NK = ro.D.shape[1]
    out = []
    for b in range(0, len(examples), bs):
        exs = examples[b:b + bs]
        if forced == 'alone':
            exs = [dict(ex, user=C.TASKS[ex['task']]) for ex in exs]
        users = [ex['user'] for ex in exs]
        bare = [{'user': C.TASKS[ex['task']], 'task': ex['task'], 'edits': []} for ex in exs]
        with torch.no_grad():
            if forced:
                mids = [C.plan_ids(tok, ex) for ex in exs]
                reply_t = [r for _, r in C.generate(model, tok, users, mids=mids)]
            else:
                reply_t, mids = C.generate(model, tok, users, think=C.PLAN), None
                if C.PLAN:
                    mids, reply_t = [m for m, _ in reply_t], [r for _, r in reply_t]
            zt, lens = own_z(lm, tok, res, ro, exs, replies, reply_t, mids)
            with model.disable_adapter():
                dz = (zt - own_z(lm, tok, res, ro, bare, replies, reply_t)[0]).cpu().numpy()
                if orig:
                    reply_o = C.generate(model, tok, users)
                    zo_, lens_o = own_z(lm, tok, res, ro, exs, replies, reply_o)
                    zo = (zo_ - own_z(lm, tok, res, ro, bare, replies, reply_o)[0]).cpu().numpy()
        T = max(lens)
        if orig:
            tgt_o = C.targets(exs, lens_o, prep, list(range(NK)), max(lens_o))[0].numpy()
        tgt, req = C.targets(exs, lens, prep, list(range(NK)), T)
        tgt, req = tgt.numpy(), req.numpy()
        for j, ex in enumerate(exs):
            n = lens[j]
            item = {'user': ex['user'], 'task': ex['task'], 'kind': ex['kind'], 'fams': ex['fams'],
                    'concepts': [ci for ci, _ in ex['edits']], 'reply': tok.decode(reply_t[j], skip_special_tokens=True)}
            if n == 0:   # the thinking block did not close: nothing done
                item.update({'frac': 0.0, 'r': 0.0, 'rel_err': 1.0, 'no_reply': True})
            else:
                item.update(item_scores(dz[j, :n], tgt[j, :n], req[j]))
            if C.PLAN and not forced:
                item.update(plan_scores(tok.decode(mids[j], skip_special_tokens=False), ex, prep['dirs'].shape[0]))
            if orig:
                no = lens_o[j]
                item['orig'] = item_scores(zo[j, :no], tgt_o[j, :no], req[j])
                item['orig_reply'] = tok.decode(reply_o[j], skip_special_tokens=True)
            if keep_profiles:
                R = item['concepts']
                item['dz'] = dz[j, :n][:, :, R].astype(np.float16)
                item['target'] = tgt[j, :n][:, :, R].astype(np.float16)
                if orig:
                    item['orig_dz'] = zo[j, :lens_o[j]][:, :, R].astype(np.float16)
            out.append(item)
    return out


def plan_scores(text, ex, L):
    """The written plan against the instruction's exact targets: plan_exact = every requested concept's plan within
    0.06 SD of its target on every layer and reply token; plan_r = correlation over the requested entries."""
    import spec
    body = text.replace('<think>', '').replace('</think>', '').strip()
    got = spec.parse_plan(body, L)
    if not ex['edits']:
        return {'plan': body, 'plan_exact': float(body == spec.plan_text([]))}
    want = np.concatenate([p.ravel() for _, p in ex['edits']])
    have = np.concatenate([got.get(K.NAMES[ci], np.zeros_like(p)).ravel() for ci, p in ex['edits']])
    r = float(np.corrcoef(want, have)[0, 1]) if have.std() > 0 and want.std() > 0 else float(np.allclose(want, have, atol=0.06))
    return {'plan': body, 'plan_exact': float(np.allclose(want, have, atol=0.06)), 'plan_r': r}


def summary(items, key, sub=None):
    xs = np.array([(it[sub] if sub else it).get(key, np.nan) for it in items], float)
    xs = xs[~np.isnan(xs)]
    if len(xs) == 0:
        return None
    rng = np.random.default_rng(0)
    bs = xs[rng.integers(0, len(xs), (2000, len(xs)))].mean(1)
    return [float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.median(xs)), len(xs)]
