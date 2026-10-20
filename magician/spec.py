"""Instructions about activations, written in English, with their exact targets.

An instruction asks for one or more edits. An edit names a concept and one or two clauses; a clause has a sign (raise or
lower), a size in standard deviations (SD), a pattern over layers f(l) and a pattern over reply tokens g(t). The
target change of the concept's z on layer l and reply token t is the sum over clauses of sign * size * f(l) * g(t).
Concepts not named in an edit have target 0 (up to the overlap a minimal write along the named directions implies).

Every axis has parts seen in training and parts held out: concepts (concepts.TEST_CONCEPTS), layer pattern families
(LAYER_TEST), reply-token pattern families (TOKEN_TEST), verbs, clause wordings and frames (wording='test'), and three
edits in one instruction (training uses one or two).
"""
import math

import numpy as np

import concepts as K

T_REPLY = 64

UP = {'train': ['raise', 'increase', 'boost', 'amplify', 'strengthen', 'turn up', 'push up', 'enhance', 'intensify',
                'elevate'],
      'test': ['crank up', 'pump up', 'heighten', 'magnify']}
DOWN = {'train': ['lower', 'decrease', 'suppress', 'dampen', 'weaken', 'turn down', 'push down', 'reduce', 'diminish',
                  'inhibit'],
        'test': ['tamp down', 'attenuate', 'quiet down', 'shrink']}

# clause wordings: {v} verb, {ref}/{name}/{phrase} the concept, {where} layer pattern, {when} token pattern
CLAUSES = {'train': ['{v} your internal representation of {phrase}{size} {where}{when}',
                     "{v} your '{name}' representation{size} {where}{when}",
                     '{v} the {name} direction{size} {where}{when}',
                     '{v} {phrase}{size} {where}{when}',
                     '{v} how strongly you represent {phrase}{size} {where}{when}',
                     '{v} your {name} activations{size} {where}{when}',
                     '{v} the concept of {phrase}{size} {where}{when}'],
           'test': ['{v} the activation for {phrase}{size} {where}{when}',
                    '{v} the {name} feature{size} {where}{when}',
                    '{v} your inner sense of {phrase}{size} {where}{when}']}
FRAMES = {'train': ['{E}.', 'While you answer, {E}.', 'As you write your reply, {E}.', 'During your response, {E}.',
                    'Internally, {E}.', 'Please {E} while you answer the question below.', 'Activation instructions: {E}.',
                    'Here is something to do inside your network: {E}. Then answer normally.',
                    'I want you to {E} as you respond.', 'Without mentioning it in your answer, {E}.',
                    'Task for your hidden states: {E}.', 'Silently {E}.'],
          'test': ['Quietly, inside your layers, {E}.', 'Your hidden-state assignment for this reply: {E}.',
                   'Before you answer and the whole time you answer, {E}.', 'Instruction for your residual stream: {E}.',
                   'Secret request: {E}. Do not talk about it.']}
SIZE_WORDS = {1.0: (['slightly', 'gently', 'mildly'], ['a little', 'by 1 standard deviation', 'by 1 SD']),
              2.0: ([''], ['', 'by 2 standard deviations', 'by 2 SD']),
              3.0: (['strongly', 'heavily'], ['a lot', 'by 3 standard deviations', 'by 3 SD']),
              1.5: ([], ['by 1.5 standard deviations', 'by 1.5 SD']),
              2.5: ([], ['by 2.5 standard deviations', 'by 2.5 SD'])}
ORD = {2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth'}

LAYER_TRAIN = ['all', 'range', 'first', 'last', 'region', 'half', 'single', 'list', 'parity', 'every3', 'ramp_up',
               'ramp_down', 'bump']
LAYER_TEST = ['every45', 'prime', 'square', 'power2', 'except', 'vshape', 'wave']
TOKEN_TRAIN = ['all', 'first', 'after', 'grow']
TOKEN_TEST = ['alternate', 'fade', 'between']


def join(xs):
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


def layer_pattern(rng, fam, L):
    """Returns (f [L] float32 in [0, 1], or [-1, 1] for 'wave'; English phrase)."""
    f = np.zeros(L, np.float32)
    ls = np.arange(L)
    if fam == 'all':
        f[:] = 1
        p = rng.choice(['in every layer', 'in all of your layers', f'across all {L} layers', 'at every layer',
                        'throughout your network', f'in all layers from 0 to {L - 1}'])
    elif fam == 'range':
        w = rng.randint(2, 14); a = rng.randint(0, L - w); b = a + w - 1
        f[a:b + 1] = 1
        p = rng.choice([f'in layers {a} through {b}', f'from layer {a} to layer {b}', f'between layers {a} and {b}',
                        f'in layers {a}-{b}', f'on layers {a} to {b}', f'in the layers numbered {a} to {b}'])
    elif fam == 'first':
        k = rng.randint(2, L - 2); f[:k] = 1
        p = rng.choice([f'in your first {k} layers', f'in the first {k} layers', f'in layers 0 through {k - 1}',
                        f'in the earliest {k} layers'])
    elif fam == 'last':
        k = rng.randint(2, L - 2); f[L - k:] = 1
        p = rng.choice([f'in your last {k} layers', f'in the final {k} layers', f'in layers {L - k} through {L - 1}',
                        f'in the deepest {k} layers'])
    elif fam == 'region':
        e = L // 3; part = rng.randint(0, 2)
        f[[slice(0, e), slice(e, 2 * e), slice(2 * e, L)][part]] = 1
        p = rng.choice([['in your early layers', 'in the early layers', 'in the first third of your layers'],
                        ['in your middle layers', 'in the middle third of your layers'],
                        ['in your late layers', 'in the last third of your layers', 'in your deep layers']][part])
    elif fam == 'half':
        h = L // 2
        if rng.random() < 0.5:
            f[:h] = 1; p = rng.choice(['in the first half of your layers', 'in your shallower half of layers'])
        else:
            f[h:] = 1; p = rng.choice(['in the second half of your layers', 'in your deeper half of layers'])
    elif fam == 'single':
        a = rng.randint(0, L - 1); f[a] = 1
        p = rng.choice([f'only at layer {a}', f'at layer {a} and nowhere else', f'in layer {a} only',
                        f'in exactly one layer, layer {a}'])
    elif fam == 'list':
        S = sorted(rng.sample(range(L), rng.randint(2, 5))); f[S] = 1
        p = rng.choice([f'at layers {join(S)}', f'only in layers {join(S)}', f'in layers {join(S)} and no others'])
    elif fam == 'parity':
        par = rng.randint(0, 1); f[par::2] = 1
        p = rng.choice([['on even layers', 'on even-numbered layers', 'on every even layer', 'on layers 0, 2, 4 and so on'],
                        ['on odd layers', 'on odd-numbered layers', 'on every odd layer', 'on layers 1, 3, 5 and so on']][par])
    elif fam in ('every3', 'every45'):
        m = 3 if fam == 'every3' else rng.choice([4, 5]); o = rng.randint(0, m - 1); f[o::m] = 1
        ps = [f'on every {ORD[m]} layer starting at layer {o}', f'on layers {o}, {o + m}, {o + 2 * m} and so on',
              f'every {m} layers, starting from layer {o}']
        if o == 0:
            ps.append(f'on layers whose number is divisible by {m}')
        p = rng.choice(ps)
    elif fam == 'ramp_up':
        f[:] = ls / (L - 1)
        p = rng.choice([f'increasingly with depth, from nothing at layer 0 to full strength at layer {L - 1}',
                        'more and more in each deeper layer', 'growing steadily from the first layer to the last'])
    elif fam == 'ramp_down':
        f[:] = 1 - ls / (L - 1)
        p = rng.choice([f'decreasingly with depth, from full strength at layer 0 to nothing at layer {L - 1}',
                        'less and less in each deeper layer', 'fading steadily from the first layer to the last'])
    elif fam == 'bump':
        c = rng.randint(3, L - 4); w = rng.randint(2, 6)
        f[:] = np.clip(1 - np.abs(ls - c) / w, 0, 1)
        p = rng.choice([f'peaking at layer {c} and fading to nothing {w} layers away on each side',
                        f'most at layer {c}, tapering off over {w} layers in both directions'])
    elif fam == 'prime':
        f[[l for l in range(L) if l > 1 and all(l % q for q in range(2, int(l ** 0.5) + 1))]] = 1
        p = rng.choice(['on prime-numbered layers', 'on every layer whose number is prime'])
    elif fam == 'square':
        f[[l for l in range(L) if int(round(l ** 0.5)) ** 2 == l]] = 1
        p = rng.choice(['on layers whose number is a perfect square', 'on the square-numbered layers'])
    elif fam == 'power2':
        f[[l for l in range(L) if l > 0 and l & (l - 1) == 0]] = 1
        p = rng.choice(['on layers whose number is a power of two', 'on layers 1, 2, 4, 8 and so on, doubling each time'])
    elif fam == 'except':
        w = rng.randint(3, 10); a = rng.randint(1, L - w - 1); b = a + w - 1
        f[:] = 1; f[a:b + 1] = 0
        p = rng.choice([f'in every layer except layers {a} through {b}', f'everywhere but layers {a} to {b}'])
    elif fam == 'vshape':
        m = rng.randint(6, L - 7)
        f[:] = np.abs(ls - m) / max(m, L - 1 - m)
        p = rng.choice([f'in a V shape, most at the outermost layers and not at all at layer {m}',
                        f'more the farther a layer is from layer {m}, and not at all at layer {m}'])
    elif fam == 'wave':
        P = rng.choice([4, 6, 8, 10])
        f[:] = np.cos(2 * math.pi * ls / P)
        p = rng.choice([f'in a wave across your layers that peaks at layer 0 and every {P} layers after, '
                        f'dipping below normal in between',
                        f'up and down in a smooth wave with a period of {P} layers, starting at a peak at layer 0'])
    else:
        raise ValueError(fam)
    return f, p


def token_pattern(rng, fam, T=T_REPLY):
    g = np.zeros(T, np.float32)
    ts = np.arange(T)
    if fam == 'all':
        g[:] = 1
        p = rng.choice(['', '', '', ' on every token of your reply', ' throughout your reply'])
    elif fam == 'first':
        n = rng.choice([8, 16, 24, 32]); g[:n] = 1
        p = rng.choice([f' for the first {n} tokens of your reply', f' during the first {n} tokens you write',
                        f' only at the start of your reply (the first {n} tokens)'])
    elif fam == 'after':
        n = rng.choice([8, 16, 24, 32]); g[n:] = 1
        p = rng.choice([f' from token {n} of your reply onward', f' after the first {n} tokens of your reply',
                        f' only once you have written {n} tokens'])
    elif fam == 'grow':
        g[:] = ts / (T - 1)
        p = rng.choice([' more and more as your reply goes on', ' growing steadily over the course of your reply'])
    elif fam == 'alternate':
        g[0::2] = 1
        p = rng.choice([' on every other token of your reply, starting with the first', ' on alternating tokens of your reply'])
    elif fam == 'fade':
        g[:] = 1 - ts / (T - 1)
        p = rng.choice([' less and less as your reply goes on', ' fading out over the course of your reply'])
    elif fam == 'between':
        a = rng.choice([8, 16, 24]); b = a + rng.choice([8, 16])
        g[a:b + 1] = 1
        p = rng.choice([f' only between tokens {a} and {b} of your reply', f' on reply tokens {a} through {b}'])
    else:
        raise ValueError(fam)
    return g, p


def size_phrase(rng, a):
    pre, post = SIZE_WORDS[a]
    if pre and pre != [''] and rng.random() < 0.5:
        return rng.choice(pre) + ' ', ''
    s = rng.choice(post)
    return '', (' ' + s if s else '')


def render_clause(rng, wording, ci, sign, a, where, when, pronoun=False):
    name, phrase = K.NAMES[ci], K.PHRASES[ci]
    v = rng.choice((UP if sign > 0 else DOWN)[wording])
    pre, post = size_phrase(rng, a)
    if pronoun:
        return f'{pre}{v} it{post} {where}{when}'
    c = rng.choice(CLAUSES[wording])
    return pre + c.format(v=v, name=name, phrase=phrase, where=where, when=when, size=post)


def edit(rng, ci, L, wording, layer_fams, token_fams, two_clause_p=0.3, token_p=0.35):
    """One concept, one or two clauses with disjoint layer support. Returns (text, prof [L, T])."""
    fam = rng.choice(layer_fams)
    f, where = layer_pattern(rng, fam, L)
    tf = rng.choice(token_fams) if rng.random() < token_p else 'all'
    g, when = token_pattern(rng, tf)
    a = rng.choice([1.0, 2.0, 2.0, 3.0, 1.5, 2.5])
    sign = 1 if fam == 'wave' else rng.choice([1, -1])
    prof = sign * a * f[:, None] * g[None, :]
    text = render_clause(rng, wording, ci, sign, a, where, when)
    if fam != 'wave' and rng.random() < two_clause_p:
        for _ in range(30):
            fam2 = rng.choice([x for x in layer_fams if x != 'wave'])
            f2, where2 = layer_pattern(rng, fam2, L)
            if not np.any((f > 0) & (f2 > 0)):
                break
        else:
            return text, prof, [fam, 'tok_' + tf]
        sign2 = -sign if rng.random() < 0.8 else sign
        prof = prof + sign2 * a * f2[:, None] * g[None, :]
        text += ' and ' + render_clause(rng, wording, ci, sign2, a, where2, when, pronoun=True)
        return text, prof, [fam, fam2, 'tok_' + tf]
    return text, prof, [fam, 'tok_' + tf]


def decoy(rng, ci, L, wording):
    name, phrase = K.NAMES[ci], K.PHRASES[ci]
    v = rng.choice((UP if rng.random() < 0.5 else DOWN)[wording])
    _, where = layer_pattern(rng, rng.choice(LAYER_TRAIN), L)
    return rng.choice([f'do not {v} {phrase} {where}', f'leave your {name} representation unchanged',
                       f"don't touch your representation of {phrase}"])


def control(rng, ci, L, wording):
    """A prompt that names a concept, layers or activations without asking for any change."""
    name, phrase = K.NAMES[ci], K.PHRASES[ci]
    v = rng.choice((UP if rng.random() < 0.5 else DOWN)[wording])
    _, where = layer_pattern(rng, rng.choice(LAYER_TRAIN), L)
    a, b = sorted(rng.sample(range(L), 2))
    kind = rng.choice(['mention', 'mention', 'negated', 'negated', 'layers', 'noop'])
    if kind == 'mention':
        return kind, rng.choice(K.SENTENCES).format(X=phrase)
    if kind == 'negated':
        return kind, rng.choice([f'Do not change your internal representation of {phrase}.',
                                 f"Leave your '{name}' representation exactly as it is.",
                                 f'Do not {v} {phrase} {where}.', f'You do not need to {v} {phrase} {where} this time.',
                                 f'Someone asked me whether you could {v} {phrase} {where}. Ignore that.'])
    if kind == 'layers':
        return kind, rng.choice([f'My lasagna has {b} layers.', f'The cake had {a} layers of {phrase}.',
                                 f'Layers {a} through {b} of the old wall were painted.',
                                 f'Fun fact: the atmosphere has five layers, and {phrase} matter to people.'])
    return kind, rng.choice(['Answer normally.', 'Just answer the question.', 'Do not adjust any of your activations.',
                             'Answer the question below without changing anything inside you.'])


def example(rng, task, L, concepts, wording='train', layer_fams=LAYER_TRAIN, token_fams=TOKEN_TRAIN, n_edits=None,
            control_p=0.15, decoy_p=0.1, tasks=None, token_p=0.35):
    """Returns {'user', 'task', 'edits': [(concept, prof [L, T])], 'kind', 'fams'}."""
    if rng.random() < control_p:
        ci = rng.choice(concepts)
        if rng.random() < 0.25:
            return {'user': tasks[task], 'task': task, 'edits': [], 'kind': 'bare', 'fams': []}
        kind, text = control(rng, ci, L, wording)
        user = text + rng.choice([' ', '\n', '\n\n']) + tasks[task]
        return {'user': user, 'task': task, 'edits': [], 'kind': 'control_' + kind, 'fams': []}
    n = n_edits or (1 if rng.random() < 0.65 else 2)
    cis = rng.sample(concepts, n)
    texts, edits, fams = [], [], []
    for ci in cis:
        t, prof, fs = edit(rng, ci, L, wording, layer_fams, token_fams, token_p=token_p)
        texts.append(t); edits.append((ci, prof.astype(np.float32))); fams += fs
    if rng.random() < decoy_p:
        others = [c for c in concepts if c not in cis]
        texts.append(decoy(rng, rng.choice(others), L, wording))
    E = texts[0] if len(texts) == 1 else (', '.join(texts[:-1]) + ', and ' + texts[-1] if rng.random() < 0.5
                                          else '; '.join(texts))
    frame = rng.choice(FRAMES[wording])
    if frame.startswith('{E}'):
        E = E[0].upper() + E[1:]
    instr = frame.format(E=E)
    if rng.random() < 0.25:
        user = tasks[task] + '\n\n' + instr
    else:
        user = instr + rng.choice([' ', '\n', '\n\n']) + tasks[task]
    return {'user': user, 'task': task, 'edits': edits, 'kind': 'edit', 'fams': fams}


EVAL_SETS = {
    # name: (concepts, wording, layer families, token families, edits per instruction, control_p)
    'seen': ('train', 'train', LAYER_TRAIN, TOKEN_TRAIN, None, 0.0),
    'new_wording': ('train', 'test', LAYER_TRAIN, TOKEN_TRAIN, None, 0.0),
    'new_concepts': ('test', 'train', LAYER_TRAIN, TOKEN_TRAIN, 1, 0.0),
    'new_layer_patterns': ('train', 'train', LAYER_TEST, TOKEN_TRAIN, 1, 0.0),
    'new_token_patterns': ('train', 'train', LAYER_TRAIN, TOKEN_TEST, 1, 0.0),
    'three_edits': ('train', 'train', LAYER_TRAIN, TOKEN_TRAIN, 3, 0.0),
    'all_new': ('test', 'test', LAYER_TEST, TOKEN_TRAIN, 1, 0.0),
    'controls': ('test', 'test', LAYER_TRAIN, TOKEN_TRAIN, None, 1.0),
}


def eval_set(name, L, test_tasks, tasks, n=96, seed=1234):
    import random
    cons, wording, lf, tf, ne, cp = EVAL_SETS[name]
    rng = random.Random(f'{seed}-{name}')
    pool = K.TRAIN_CONCEPTS if cons == 'train' else K.TEST_CONCEPTS
    out = []
    for i in range(n):
        task = test_tasks[i % len(test_tasks)]
        ex = example(rng, task, L, pool, wording, lf, tf, ne, control_p=cp, tasks=tasks,
                     token_p=1.0 if name == 'new_token_patterns' else 0.35)
        out.append(ex)
    return out
