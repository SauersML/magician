"""Everything training needs from the original model, written to OUT/prep.pt:

1. its greedy replies (64 tokens) to the 310 questions, with no instruction (OUT/replies.json);
2. one unit direction per concept and layer: the mean residual over the concept's slot span (from the concept to the
   end of the sentence) across 40 sentences, minus that mean averaged over the training concepts; the other 8
   sentences, minus the same average, check that each concept's own direction scores it highest (accuracy per layer);
3. m, s: mean and SD of every concept's projection over the reply tokens of the training questions;
4. the original model's top-64 next-token log-probabilities on every reply token (for the KL term);
5. gram: dot products between the unit directions at each layer.
"""
import json
import os
import time

import torch

import common as C
import concepts as K

t0 = time.time()
tok, model = C.load()
res = C.Residuals(model)
L = len(res.layers)
d = model.get_input_embeddings().embedding_dim
NK = len(K.CONCEPTS)

# 1. replies
rp = os.path.join(C.OUT, 'replies.json')
if not os.path.exists(rp):
    tok.padding_side = 'left'
    replies = []
    with torch.no_grad():
        for b in range(0, len(C.TASKS), 32):
            prompts = [C.chat_prefix(tok, t) for t in C.TASKS[b:b + 32]]
            enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(C.DEVICE)
            g = model.generate(**enc, max_new_tokens=C.REPLY_TOKENS, do_sample=False)
            for row in g[:, enc.input_ids.shape[1]:].tolist():
                cut = next((i + 1 for i, x in enumerate(row) if x in C.end_ids(tok)), len(row))
                replies.append(row[:cut])
    tok.padding_side = 'right'
    json.dump([{'task': t, 'ids': r, 'text': tok.decode(r)} for t, r in zip(C.TASKS, replies)], open(rp, 'w'), indent=0)
    print(f'replies {time.time() - t0:.0f}s', flush=True)
replies = json.load(open(rp))

# 2. concept directions
prefix = C.chat_prefix(tok, 'Write one sentence.')
DIR_S, VAL_S = (K.SENTENCES[:3], K.SENTENCES[40:41]) if C.SMOKE else (K.SENTENCES[:40], K.SENTENCES[40:])


def span_means(sentences, average=False):
    """[len(sentences), NK, L, d] mean residual over each concept's slot span ([NK, L, d] averaged over sentences)."""
    items = []
    for si, s in enumerate(sentences):
        for k, phrase in enumerate(K.PHRASES):
            text = prefix + s.format(X=phrase)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            start = len(prefix) + s.index('{X}')
            a = next(i for i, (x, y) in enumerate(enc.offset_mapping) if y > start)
            items.append((si, k, enc.input_ids, a))
    out = torch.zeros(1 if average else len(sentences), NK, L, d)
    with torch.no_grad():
        for b in range(0, len(items), 128):
            chunk = items[b:b + 128]
            ids, att = C.pad_right([x[2] for x in chunk], tok.pad_token_id)
            model(input_ids=ids.to(C.DEVICE), attention_mask=att.to(C.DEVICE))
            for j, (si, k, seq, a) in enumerate(chunk):
                out[0 if average else si, k] += torch.stack([res.h[l][j, a:len(seq)].float().mean(0) for l in range(L)]).cpu()
    return out[0] / len(sentences) if average else out


M = span_means(DIR_S, average=True)                                    # [NK, L, d]
center = M[K.TRAIN_CONCEPTS].mean(0)
dirs = M - center
dirs = dirs / dirs.norm(dim=-1, keepdim=True)
dirs = dirs.permute(1, 0, 2).contiguous()                              # [L, NK, d]
V = span_means(VAL_S)                                                  # [8, NK, L, d]
sc = torch.einsum('sknd,nqd->sknq', V - center, dirs)                  # [8, NK, L, NK], centred like the directions
acc = (sc.argmax(-1) == torch.arange(NK)[None, :, None]).float()       # [8, NK, L]
val_acc = {'train': acc[:, K.TRAIN_CONCEPTS].mean((0, 1)).tolist(), 'test': acc[:, K.TEST_CONCEPTS].mean((0, 1)).tolist()}
print(f'directions {time.time() - t0:.0f}s; held-out sentence accuracy (180-way) per layer, test concepts:',
      ' '.join(f'{x:.2f}' for x in val_acc['test']), flush=True)
del V, sc

# 3-4. statistics and KL cache on the original model's own replies
tr_t, te_t = C.splits()
Dd = dirs.to(C.DEVICE)
S1 = torch.zeros(L, NK, dtype=torch.float64); S2 = torch.zeros_like(S1); N = 0
kl = [None] * len(C.TASKS)
with torch.no_grad():
    for b in range(0, len(C.TASKS), 16):
        idx = list(range(b, min(b + 16, len(C.TASKS))))
        seqs, starts = [], []
        for t in idx:
            p = tok(C.chat_prefix(tok, C.TASKS[t]), add_special_tokens=False).input_ids
            seqs.append(p + replies[t]['ids']); starts.append(len(p))
        ids, att = C.pad_right(seqs, tok.pad_token_id)
        hs = model.model(input_ids=ids.to(C.DEVICE), attention_mask=att.to(C.DEVICE)).last_hidden_state
        for j, t in enumerate(idx):
            a, n = starts[j], len(replies[t]['ids'])
            lp = torch.log_softmax(model.lm_head(hs[j, a - 1:a + n - 1]).float(), -1)
            v, i = lp.topk(64, -1)
            kl[t] = {'topv': v.half().cpu(), 'topi': i.int().cpu()}
            if t in tr_t:
                P = torch.stack([res.h[l][j, a:a + n].float() @ Dd[l].T for l in range(L)], 1).cpu().double()  # [n, L, NK]
                S1 += P.sum(0); S2 += (P ** 2).sum(0); N += n
m = (S1 / N).float()
s = ((S2 / N - (S1 / N) ** 2).clamp_min(1e-12).sqrt()).float()
gram = torch.einsum('lkd,lqd->lkq', dirs, dirs)
torch.save({'dirs': dirs, 'm': m, 's': s, 'gram': gram, 'kl': kl, 'val_acc': val_acc, 'model': C.MODEL,
            'train_concepts': K.TRAIN_CONCEPTS, 'test_concepts': K.TEST_CONCEPTS}, os.path.join(C.OUT, 'prep.pt'))
off = gram.clone(); off[:, torch.arange(NK), torch.arange(NK)] = 0
print(f'prep done {time.time() - t0:.0f}s; reply tokens {N}; mean |cos| between concept directions per layer:',
      ' '.join(f'{x:.2f}' for x in off.abs().mean((1, 2)).tolist()), flush=True)
