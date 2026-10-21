"""Model loading, chat formatting, residual capture, concept readouts and targets, shared by prep, train and evaluate.

z_{k,l,t} = (h_{l,t} . c_{k,l} - m_{k,l}) / s_{k,l}: the residual after block l on reply token t, projected on concept
k's unit direction at layer l, standardized by the mean m and SD s of that projection over the original model's
reply tokens to the training questions (no instruction). Training reads the original model's reply (teacher forcing);
evaluation measures each model on the reply it writes itself (measure.py).
"""
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('MAGIC_OUT', os.path.join(os.path.dirname(ROOT), 'runs'))
MODEL = os.environ.get('MAGIC_MODEL', 'Qwen/Qwen3-1.7B')
DEVICE = os.environ.get('MAGIC_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = os.environ.get('MAGIC_SMOKE') == '1'  # a few questions and sentences, to test the code end to end
TASKS = json.load(open(os.path.join(ROOT, 'tasks.json')))[:40 if SMOKE else None]
REPLY_TOKENS = 64
PLAN = os.environ.get('MAGIC_PLAN', '0') in ('1', '2')  # write an explicit plan in the thinking block, then answer (2: as a table)
MAX_PLAN = int(os.environ.get('MAGIC_MAX_PLAN', '768'))
os.makedirs(OUT, exist_ok=True)


def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEVICE).eval()
    return tok, model


def think_end(tok):
    return tok.convert_tokens_to_ids('</think>')


def end_ids(tok):
    """Tokens that end a reply."""
    return {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')}


def chat_prefix(tok, user, think=False):
    """think=False: the prefix ends with an empty thinking block; think=True: the model writes its own."""
    return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=think)


def plan_ids(tok, ex):
    """The thinking block holding ex['plan'] (a plan written elsewhere), or else the exact plan for ex's edits."""
    import spec
    text = ex.get('plan') or spec.plan_text(ex['edits'])
    return tok('<think>\n' + text + '\n</think>\n\n', add_special_tokens=False).input_ids


def splits(seed=0):
    """75/25 split of the questions (the same split as the zigzag repo)."""
    rng = random.Random(seed)
    t = list(range(len(TASKS))); rng.shuffle(t)
    nt = int(0.75 * len(t))
    return sorted(t[:nt]), sorted(t[nt:])


def find_layers(model):
    """The decoder blocks (of the language model, in a model that also reads images)."""
    for name, mod in model.named_modules():
        if name.endswith('layers') and isinstance(mod, torch.nn.ModuleList) and 'visual' not in name:
            return mod


class Residuals:
    """Records the residual stream after every decoder block."""

    def __init__(self, model):
        self.layers = find_layers(model)
        self.h = [None] * len(self.layers)
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(self.layers)]

    def _hook(self, i):
        def f(mod, inp, out):
            self.h[i] = out[0] if isinstance(out, tuple) else out
        return f


def pad_right(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1
    return ids, mask


def kl_topk(base_v, base_i, logits):
    """KL(original || trained) over the original model's top-k tokens plus one bucket for the rest."""
    lp = torch.log_softmax(logits.float(), -1)
    t = lp.gather(-1, base_i)
    pb = base_v.exp()
    rest_b = (1 - pb.sum(-1)).clamp_min(1e-8)
    rest_t = (1 - t.exp().sum(-1)).clamp_min(1e-8)
    return (pb * (base_v - t)).sum(-1) + rest_b * (rest_b.log() - rest_t.log())


class Readout:
    """Concept z on reply tokens for a batch, from the residuals a forward pass recorded."""

    def __init__(self, prep, cols=None):
        self.cols = torch.arange(prep['dirs'].shape[1]) if cols is None else torch.as_tensor(cols)
        self.D = prep['dirs'][:, self.cols].float().to(DEVICE)          # [L, K, d]
        self.m = prep['m'][:, self.cols].float().to(DEVICE)             # [L, K]
        self.s = prep['s'][:, self.cols].float().to(DEVICE)

    def z(self, hs, pos, mask):
        """hs: L tensors [B, S, d]; pos, mask: [B, T] reply positions. Returns [B, T, L, K]."""
        idx = pos.to(DEVICE)[..., None].expand(-1, -1, hs[0].shape[-1])
        out = []
        for l, h in enumerate(hs):
            g = h.gather(1, idx).float()                               # [B, T, d]
            out.append((g @ self.D[l].T - self.m[l]) / self.s[l])
        return torch.stack(out, 2) * mask.to(DEVICE)[..., None, None]


def generate(model, tok, users, bs=32, think=False, mids=None):
    """Greedy replies (token ids, up to and including the end token), REPLY_TOKENS at most, as prep.py writes them.
    think=True: the model first writes its thinking block (MAX_PLAN tokens at most); returns (thinking block ids, reply
    ids) pairs, with an empty reply when the block does not close. mids: given thinking blocks (token ids, as plan_ids
    writes them) after each prompt; the model writes only the reply, and the result is (mids[j], reply) pairs."""
    tok.padding_side = 'left'
    out = []
    te, blank, ends = think_end(tok), tok('\n\n', add_special_tokens=False).input_ids, end_ids(tok)
    with torch.no_grad():
        for b in range(0, len(users), bs):
            if mids is None:
                enc = tok([chat_prefix(tok, u, think) for u in users[b:b + bs]], return_tensors='pt', padding=True,
                          add_special_tokens=False).to(DEVICE)
                ids, att = enc.input_ids, enc.attention_mask
            else:
                P = [tok(chat_prefix(tok, u, True), add_special_tokens=False).input_ids + m
                     for u, m in zip(users[b:b + bs], mids[b:b + bs])]
                T = max(len(p) for p in P)
                ids = torch.tensor([[tok.pad_token_id] * (T - len(p)) + p for p in P], device=DEVICE)
                att = torch.tensor([[0] * (T - len(p)) + [1] * len(p) for p in P], device=DEVICE)
            g = model.generate(input_ids=ids, attention_mask=att, do_sample=False,
                               max_new_tokens=REPLY_TOKENS + (MAX_PLAN if think and mids is None else 0))
            for j, row in enumerate(g[:, ids.shape[1]:].tolist()):
                mid = []
                if mids is not None:
                    mid = mids[b + j]
                elif think:
                    if te not in row:
                        out.append((row, []))
                        continue
                    k = row.index(te) + 1
                    k += len(blank) if row[k:k + len(blank)] == blank else 0      # the blank line after </think>
                    mid, row = row[:k], row[k:k + REPLY_TOKENS]
                cut = next((i + 1 for i, x in enumerate(row) if x in ends), len(row))
                out.append((mid, row[:cut]) if think or mids is not None else row[:cut])
    tok.padding_side = 'right'
    return out


def batch_inputs(tok, examples, replies, own=None, mids=None):
    """Token ids of instruction prompt + reply (the original model's reply to the bare question, or own[j]); with
    mids, the prompt opens the thinking block and mids[j] (thinking block ids) comes before the reply. Reply positions
    [B, T] and their mask."""
    seqs, starts, lens = [], [], []
    for j, ex in enumerate(examples):
        p = tok(chat_prefix(tok, ex['user'], mids is not None), add_special_tokens=False).input_ids
        p = p + (mids[j] if mids is not None else [])
        r = replies[ex['task']]['ids'] if own is None else own[j]
        seqs.append(p + r); starts.append(len(p)); lens.append(len(r))
    ids, att = pad_right(seqs, tok.pad_token_id)
    T = max(lens)
    pos = torch.zeros(len(seqs), T, dtype=torch.long)
    mask = torch.zeros(len(seqs), T)
    for j, (a, n) in enumerate(zip(starts, lens)):
        pos[j, :n] = torch.arange(a, a + n); pos[j, n:] = a + n - 1
        mask[j, :n] = 1
    return ids, att, pos, mask, starts, lens


def targets(examples, lens, prep, cols, T):
    """Target change [B, T, L, K] on the readout's concepts: the requested profiles, and on every other concept the
    change that the smallest residual write reaching the requested profiles would cause (through the overlap of the
    concept directions). Also a mask [B, K] of requested concepts."""
    G, s = prep['gram'].numpy(), prep['s'].numpy()                    # [L, K, K], [L, K]
    cols = np.asarray(cols)
    L = G.shape[0]
    out = np.zeros((len(examples), T, L, len(cols)), np.float32)
    req = np.zeros((len(examples), len(cols)), np.float32)
    where = {c: i for i, c in enumerate(cols)}
    for j, ex in enumerate(examples):
        if not ex['edits']:
            continue
        R = [ci for ci, _ in ex['edits']]
        n = lens[j]
        dz = np.stack([prof[:, :n] for _, prof in ex['edits']], 0)    # [R, L, n]
        for l in range(L):
            GRR = G[l][np.ix_(R, R)]
            coef = np.linalg.solve(GRR, s[l, R][:, None] * dz[:, l, :])           # [R, n]
            imp = (G[l][np.ix_(cols, R)] @ coef) / s[l, cols][:, None]            # [K, n]
            out[j, :n, l, :] = imp.T
        for ci in R:
            if ci in where:
                req[j, where[ci]] = 1
    return torch.from_numpy(out), torch.from_numpy(req)
