"""Plans written by the original model (the adapter off), from a few training examples, for the trained model to carry
out. The plan text follows spec.plan_text. Decoding is greedy with two rules: each line opens with a real concept's
name, and every layer list keeps to existing layers in increasing order (when the model starts a layer past the last
one, or one at or below the layer before it, the ', ' is taken back, the list ends with ';' and decoding goes on)."""
import contextlib
import copy
import random
import re

import torch

import common as C
import concepts as K
import spec


def instruction(ex):
    return ex['user'].replace(C.TASKS[ex['task']], '').strip()


def prefix_text(L, n_shots=16, seed=7):
    rng = random.Random(seed)
    tr_t, _ = C.splits()
    shots = [spec.example(rng, rng.choice(tr_t), L, K.TRAIN_CONCEPTS, 'train', spec.LAYER_MIX, spec.TOKEN_TRAIN,
                          n_edits=1, control_p=0.0, decoy_p=0.0, tasks=C.TASKS) for _ in range(n_shots)]
    head = (f'Each instruction asks a language model to change its internal representation of a concept. The model has '
            f'{L} layers, numbered 0 to {L - 1}, and the reply has {spec.T_REPLY} tokens, numbered 0 to '
            f'{spec.T_REPLY - 1}. The plan states the change in SD on every layer it touches and the reply tokens it '
            'applies to.\n\n')
    return head + ''.join(f'Instruction: {instruction(ex)}\nPlan:\n{spec.plan_text(ex["edits"])}\n\n' for ex in shots)


def bad_at(text, L):
    """Index of the ', ' before the first layer in the current line that is past the last layer or not above the
    layer before it; None if every layer so far is fine."""
    start = text.rfind('\n') + 1
    seg = text[start:]
    for m in re.finditer(r'SD on layers ([0-9, ]*)|layer by layer ((?:\d+ [+-]?[0-9.]*(?:, ?)?)*)', seg):
        body, base = (m.group(1), m.start(1)) if m.group(1) is not None else (m.group(2), m.start(2))
        prev, pos = -1, 0
        for item in body.split(', '):
            num = item.split(' ')[0].rstrip(',')
            # complete when anything follows it: a later ', ', its own ',', its value, or the text after the list
            done = pos + len(item) < len(body) or item.endswith(',') or m.end() < len(seg) or ' ' in item.strip()
            if num.isdigit() and (int(num) >= L or (done and int(num) <= prev)):
                return start + base + pos - 2 if pos >= 2 else None
            if num.isdigit():
                prev = int(num)
            pos += len(item) + 2
    return None


class Planner:
    def __init__(self, model, tok, L):
        self.model, self.tok, self.L = model, tok, L
        self.off = model.disable_adapter if hasattr(model, 'disable_adapter') else contextlib.nullcontext
        self.pids = tok(prefix_text(L), return_tensors='pt').input_ids.to(C.DEVICE)
        with torch.no_grad(), self.off():
            self.cache = model(self.pids, use_cache=True).past_key_values

    def _name_token(self, logits, line):
        """The most likely next token that keeps the line's opening on a real concept's name followed by ': '."""
        tok, w0 = self.tok, line.lstrip()
        cands = {tok(r, add_special_tokens=False).input_ids[0] for r in
                 ((nm + ': ')[len(w0):] for nm in K.NAMES if (nm + ': ').startswith(w0)) if r}
        for t in torch.topk(logits, 200).indices.tolist():
            w = (line + tok.decode([t])).lstrip()
            if any((nm + ': ').startswith(w) or w.startswith(nm + ': ') for nm in K.NAMES):
                cands.add(t)
                break
        return max(cands, key=lambda t: float(logits[t])) if cands else int(logits.argmax())

    def __call__(self, ex, max_new=None):
        tok, model = self.tok, self.model
        max_new = max_new or 60 + 7 * self.L
        cache = copy.deepcopy(self.cache)
        q = tok(f'Instruction: {instruction(ex)}\nPlan:\n', add_special_tokens=False, return_tensors='pt').input_ids
        base = self.pids.shape[1] + q.shape[1]
        gen = []
        with torch.no_grad(), self.off():
            logits = model(q.to(C.DEVICE), past_key_values=cache, use_cache=True).logits[0, -1]
            for _ in range(max_new):
                line = tok.decode(gen).rpartition('\n')[2]
                nxt = self._name_token(logits, line) if ': ' not in line else int(logits.argmax())
                text = tok.decode(gen + [nxt])
                cut = bad_at(text, self.L)
                feed = [nxt]
                if cut is not None:
                    k = max(i for i in range(len(gen) + 1) if len(tok.decode(gen[:i])) <= cut)
                    feed = tok(text[len(tok.decode(gen[:k])):cut] + ';', add_special_tokens=False).input_ids
                    gen = gen[:k]
                    cache.crop(base + k)
                gen += feed
                if '\n\n' in tok.decode(gen) or tok.eos_token_id in gen:
                    break
                logits = model(torch.tensor([feed], device=C.DEVICE), past_key_values=cache, use_cache=True).logits[0, -1]
        return tok.decode(gen, skip_special_tokens=True).split('\n\n')[0].strip()
