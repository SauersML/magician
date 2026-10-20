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


def prefix_text(L, seed=7):
    """The header (the instruction language's size words and the rule for concepts left alone) and one example per
    training layer pattern, plus two with a concept the instruction leaves alone. Held-out patterns appear nowhere."""
    rng = random.Random(seed)
    tr_t, _ = C.splits()
    fams = list(dict.fromkeys(spec.LAYER_MIX))
    shots = [spec.example(rng, rng.choice(tr_t), L, K.TRAIN_CONCEPTS, 'train', [f], spec.TOKEN_TRAIN, n_edits=1,
                          control_p=0.0, decoy_p=0.0, tasks=C.TASKS) for f in fams]
    shots += [spec.example(rng, rng.choice(tr_t), L, K.TRAIN_CONCEPTS, 'train', spec.LAYER_MIX, spec.TOKEN_TRAIN,
                           n_edits=1, control_p=0.0, decoy_p=1.0, tasks=C.TASKS) for _ in range(2)]
    rng.shuffle(shots)
    words = lambda a: ', '.join(w for w in spec.SIZE_WORDS[a][0] + spec.SIZE_WORDS[a][1][:1] if w)
    head = (f'Each instruction asks a language model to change its internal representation of a concept. The model has '
            f'{L} layers, numbered 0 to {L - 1}, and the reply has {spec.T_REPLY} tokens, numbered 0 to '
            f'{spec.T_REPLY - 1}. The plan states the change in SD on every layer it touches and the reply tokens it '
            f'applies to: + to raise, - to lower. With no number given, {words(1.0)} mean 1 SD, {words(3.0)} mean 3 SD, '
            'and no size word means 2 SD. A concept the instruction says to leave unchanged gets no line.\n\n')
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
    """planner(ex) writes one plan straight after the examples. With think=True, planner.plans(exs) first lets the model
    think (greedy, at most think_tokens, all instructions in batches), then writes each plan under the same rules."""

    def __init__(self, model, tok, L, think=False, think_tokens=1024):
        self.model, self.tok, self.L, self.think, self.think_tokens = model, tok, L, think, think_tokens
        self.off = model.disable_adapter if hasattr(model, 'disable_adapter') else contextlib.nullcontext
        self.prefix = prefix_text(L)
        self.pids = tok(self.prefix, return_tensors='pt').input_ids.to(C.DEVICE)
        if not think:
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

    def _decode(self, cache, base, logits, max_new=None):
        """Greedy plan text from the next-token logits after `base` cached tokens, under the name and layer rules."""
        tok, model = self.tok, self.model
        gen = []
        for _ in range(max_new or 60 + 7 * self.L):
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

    def __call__(self, ex):
        if self.think:
            return self.plans([ex])[0]
        cache = copy.deepcopy(self.cache)
        q = self.tok(f'Instruction: {instruction(ex)}\nPlan:\n', add_special_tokens=False, return_tensors='pt').input_ids
        with torch.no_grad(), self.off():
            logits = self.model(q.to(C.DEVICE), past_key_values=cache, use_cache=True).logits[0, -1]
            return self._decode(cache, self.pids.shape[1] + q.shape[1], logits)

    def plans(self, exs, bs=16):
        if not self.think:
            return [self(ex) for ex in exs]
        tok, model = self.tok, self.model
        prompts = [tok(C.chat_prefix(tok, self.prefix + f'Instruction: {instruction(ex)}\nWrite the plan for this last '
                                     'instruction in exactly the format of the examples.', think=True),
                       add_special_tokens=False).input_ids for ex in exs]
        out = []
        tok.padding_side = 'left'
        with torch.no_grad(), self.off():
            for b in range(0, len(prompts), bs):
                P = prompts[b:b + bs]
                T = max(len(p) for p in P)
                ids = torch.tensor([[tok.pad_token_id] * (T - len(p)) + p for p in P], device=C.DEVICE)
                att = torch.tensor([[0] * (T - len(p)) + [1] * len(p) for p in P], device=C.DEVICE)
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=self.think_tokens)
                for p, row in zip(P, g[:, T:].tolist()):
                    # the thinking up to </think> (closed by hand if the budget ran out), then the plan under the rules
                    th = row[:row.index(C.THINK_END) + 1] if C.THINK_END in row else \
                        [t for t in row if t != tok.pad_token_id] + tok('\n</think>', add_special_tokens=False).input_ids
                    full = p + th + tok('\n\n', add_special_tokens=False).input_ids
                    o = model(torch.tensor([full], device=C.DEVICE), use_cache=True)
                    out.append(self._decode(o.past_key_values, len(full), o.logits[0, -1]))
        tok.padding_side = 'right'
        return out
