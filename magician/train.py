"""LoRA-train the model to carry out instructions about its own activations (spec.example, training splits only).

Per reply token and layer, with dz the trained model's z minus the original model's z on the same prompt:
  requested concepts: MSE(dz, target), with the entries where the target is nonzero and the entries where it is zero
                      weighted equally per example and concept (--balance 1), so a sparse pattern counts in full;
  every other training concept: MSE(dz, the change a smallest write of the requested change implies), weight beta;
  output: KL(original model on the bare question || trained model on the instruction prompt), weight lam;
  with MAGIC_PLAN=1, the thinking block holds the exact plan (spec.plan_text) before the reply, with a next-token loss
  on its tokens, weight --plan_w.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model

import common as C
import concepts as K
import measure
import spec

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='m1')
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--lam', type=float, default=3.0)
ap.add_argument('--lam_ramp', type=int, default=200)
ap.add_argument('--beta', type=float, default=1.0)
ap.add_argument('--eval_every', type=int, default=250)
ap.add_argument('--eval_first', type=int, default=1, help='0: no eval at step 0 (with plans, a fresh adapter '
                'never closes its thinking block, so the step 0 eval only takes time)')
ap.add_argument('--save_every', type=int, default=1000)
ap.add_argument('--eval_n', type=int, default=24)
ap.add_argument('--init', default='')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--grad_ckpt', type=int, default=0)
ap.add_argument('--balance', type=int, default=1)
ap.add_argument('--plan_w', type=float, default=1.0)
ap.add_argument('--plan_only_p', type=float, default=0.0, help='with plans: share of edit examples whose prompt is the '
                'bare question, so the plan in the thinking block is the only place the edit is written (no plan '
                'loss on them)')
args = ap.parse_args()

torch.manual_seed(args.seed)
rng = random.Random(args.seed)
prep = torch.load(os.path.join(C.OUT, 'prep.pt'), weights_only=False)
replies = json.load(open(os.path.join(C.OUT, 'replies.json')))
tok, model = C.load()
if args.grad_ckpt:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
if args.init:
    model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj',
                                                             'up_proj', 'down_proj']))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
lm = model.base_model.model
res = C.Residuals(model)
L = prep['dirs'].shape[0]
bank = K.TRAIN_CONCEPTS
ro = C.Readout(prep, bank)
tr_t, te_t = C.splits()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))
evals = {n: spec.eval_set(n, L, te_t, C.TASKS, n=args.eval_n, seed=99)
         for n in ('seen', 'seen_new_blocks', 'new_concepts', 'new_layer_patterns', 'new_token_patterns', 'controls')}
odir = os.path.join(C.OUT, args.name)
os.makedirs(odir, exist_ok=True)


def evaluate(step):
    model.eval()
    row = {'step': step}
    for n, exs in evals.items():
        items = measure.score(lm, model, tok, res, prep, exs, replies, orig=(step == 0))
        key = 'rms' if n == 'controls' else 'frac'
        row[n] = {k: measure.summary(items, k) for k in ('frac', 'r', 'rel_err', 'leak', 'rms', 'plan_exact', 'plan_r')}
        if step == 0:
            row[n]['orig'] = {k: measure.summary(items, k, 'orig') for k in ('frac', 'r', 'rms')}
    if C.PLAN:   # the exact plan after the bare question: does the model carry out a plan it did not get from the prompt?
        for n in ('seen', 'new_layer_patterns'):
            row[n]['alone_r'] = measure.summary(measure.score(lm, model, tok, res, prep, evals[n], replies, forced='alone'), 'r')
    model.train()
    f = lambda s, k: (row[s][k] or [float('nan')])[0]
    print(f'EVAL step {step}: ' + '  '.join(f'{n} frac {f(n, "frac"):+.2f} r {f(n, "r"):+.2f} leak {f(n, "leak"):.2f}'
                                          + (f' plan {f(n, "plan_exact"):.2f}' if C.PLAN else '') for n in evals if n != 'controls')
          + f'  controls rms {f("controls", "rms"):.3f} leak {f("controls", "leak"):.3f}'
          + (f'  plan alone r: seen {f("seen", "alone_r"):+.2f} new_layer_patterns {f("new_layer_patterns", "alone_r"):+.2f}'
             if C.PLAN else ''), flush=True)
    return row


log = []
t0 = time.time()
model.train()
for step in range(args.steps + 1):
    if (step % args.eval_every == 0 and (step > 0 or args.eval_first)) or step == args.steps:
        log.append(evaluate(step))
        json.dump({'args': vars(args), 'log': log}, open(os.path.join(odir, 'log.json'), 'w'), indent=1)
        if step > 0 and (step % args.save_every == 0 or step == args.steps):
            model.save_pretrained(os.path.join(odir, f'step_{step}'))
    if step == args.steps:
        break
    exs = [spec.example(rng, rng.choice(tr_t), L, K.TRAIN_CONCEPTS, 'train', spec.LAYER_MIX, spec.TOKEN_TRAIN,
                        tasks=C.TASKS) for _ in range(args.bs)]
    if C.PLAN and args.plan_only_p:
        exs = [dict(ex, user=C.TASKS[ex['task']], plan_only=True) if ex['edits'] and rng.random() < args.plan_only_p
               else ex for ex in exs]
    mids = [C.plan_ids(tok, ex) for ex in exs] if C.PLAN else None
    ids, att, pos, mask, starts, lens = C.batch_inputs(tok, exs, replies, mids=mids)
    T = pos.shape[1]
    tgt, req = C.targets(exs, lens, prep, bank, T)
    tgt, req, maskd = tgt.to(C.DEVICE), req.to(C.DEVICE), mask.to(C.DEVICE)
    with torch.no_grad(), model.disable_adapter():
        measure.forward(lm, ids, att)
        zb = ro.z(res.h, pos, mask)
    hs = measure.forward(lm, ids, att)
    dz = ro.z(res.h, pos, mask) - zb
    err = (dz - tgt) ** 2                                              # [B, T, L, K], zero on padding
    w_req = req[:, None, None, :] * maskd[..., None, None]
    w_bank = (1 - req[:, None, None, :]) * maskd[..., None, None]
    if args.balance:
        on = (tgt.abs() > 1e-6).float() * w_req                       # [B, T, L, K]
        off = (tgt.abs() <= 1e-6).float() * w_req
        n_on, n_off = on.sum((1, 2)), off.sum((1, 2))                  # [B, K]
        per = ((err * on).sum((1, 2)) / n_on.clamp_min(1) + (err * off).sum((1, 2)) / n_off.clamp_min(1)) \
            / ((n_on > 0).float() + (n_off > 0).float()).clamp_min(1)
        l_req = (per * req).sum() / req.sum().clamp_min(1)
    else:
        l_req = (err * w_req).sum() / (w_req.sum() * L).clamp_min(1)
    l_bank = (err * w_bank).sum() / (w_bank.sum() * L).clamp_min(1)
    g = hs.gather(1, (pos.to(C.DEVICE) - 1)[..., None].expand(-1, -1, hs.shape[-1]))
    topv = torch.zeros(len(exs), T, 64, device=C.DEVICE)
    topi = torch.zeros(len(exs), T, 64, dtype=torch.long, device=C.DEVICE)
    for j, ex in enumerate(exs):
        k = prep['kl'][ex['task']]
        topv[j, :lens[j]] = k['topv'].float().to(C.DEVICE); topi[j, :lens[j]] = k['topi'].long().to(C.DEVICE)
    kl = (C.kl_topk(topv, topi, lm.lm_head(g)) * maskd).sum() / maskd.sum()
    lam = args.lam * min(1.0, (step + 1) / args.lam_ramp)
    loss = l_req + args.beta * l_bank + lam * kl
    l_plan = torch.zeros((), device=C.DEVICE)
    if C.PLAN:
        written = [j for j, ex in enumerate(exs) if not ex.get('plan_only')]
        for j in written:
            a1 = starts[j]; a0 = a1 - len(mids[j])
            l_plan = l_plan + torch.nn.functional.cross_entropy(lm.lm_head(hs[j, a0 - 1:a1 - 1]).float(),
                                                                ids[j, a0:a1].to(C.DEVICE)) / len(written)
        loss = loss + args.plan_w * l_plan
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 10 == 0:
        with torch.no_grad():
            num = (dz * tgt * w_req).sum(); den = (tgt ** 2 * w_req).sum().clamp_min(1e-9)
        print(f'step {step:5d} {time.time() - t0:6.0f}s loss {loss.item():.3f} req {l_req.item():.3f} '
              f'bank {l_bank.item():.4f} kl {kl.item():.4f} plan {l_plan.item():.4f} batch frac {(num / den).item():+.2f}',
              flush=True)
