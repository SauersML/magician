"""Coverage of the confidence intervals by simulation: how often each 95% interval contains its target.

Populations whose target is known:
  observed   the 96 per-instruction scores of one test set and arm (results/eval_r4.json) as the population, drawn
             either independently from all 96 (same), or as 12 strata of 8 sorted scores, 8 draws from each (strata:
             independent instructions with very different distributions, the stress case for the question strata of
             spec.eval_set);
  synthetic  heavy-tailed, skewed, outlier, bounded skewed, and 0/1 scores, with and without strata, and a population
             in which 8 of the 96 instructions carry all the skew (groups: 12 groups of 8 instructions, the
             last group drawn from a skewed distribution and the others from a narrow one).
Methods:
  bootstrap  the percentile bootstrap of the mean used before (2000 resamples), for the mean;
  median     intervals.median_ci, for the median of the average distribution function;
  exact      intervals.binary_ci, for the mean of a 0/1 score.
Writes results/interval_coverage.json: one entry per population, design, and method with the coverage over REPS
samples of n = 96 and the mean interval width.

Usage: python interval_coverage.py   (from magician/; no model, no GPU)
"""
import json
import os

import numpy as np
from scipy.stats import quantile_test

import intervals

N, REPS, B = 96, 2000, 2000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(20261020)

# the order statistics of the exact median interval depend only on n: take them from scipy once
_x = np.arange(N, dtype=float)
_ci = quantile_test(_x, p=0.5).confidence_interval(confidence_level=1 - intervals.ALPHA)
K1, K2 = int(_ci.low), int(_ci.high)            # 0-based ranks of the lower and upper order statistics


def draw(pop, design, reps):
    """reps samples of N scores: same = independent draws from pop; strata = 12 strata of sorted pop, N // 12 draws
    from each; groups = the same with 12 contiguous blocks of pop, unsorted."""
    pop = np.asarray(pop, float)
    if design == 'same':
        return pop[rng.integers(0, len(pop), (reps, N))]
    s = (np.sort(pop) if design == 'strata' else pop).reshape(12, -1)
    k = N // 12
    idx = rng.integers(0, s.shape[1], (reps, 12, k))
    return np.take_along_axis(np.broadcast_to(s, (reps, 12, s.shape[1])), idx, 2).reshape(reps, N)


def cover_median(X, pop):
    """Coverage of the exact median interval (the sample's order statistics K1 and K2) for the median of the average
    distribution function, which is that of pop for both designs (equal strata). With an even number of distinct
    values, both middle values are medians: the coverage reported is the smaller of the two."""
    pop = np.asarray(pop, float)
    S = np.sort(X, 1)
    a, b = S[:, K1], S[:, K2]
    cov = min(float(np.mean((a <= q) & (q <= b))) for q in (np.quantile(pop, 0.5, method='inverted_cdf'),
                                                             np.quantile(pop, 0.5, method='higher')))
    return cov, float(np.mean(b - a))


def cover_bootstrap(X, target):
    """The old interval: percentile bootstrap of the mean, B resamples. One draw of resampling counts serves every
    sample (each sample's interval is still the method's interval; only the coverage estimate's noise changes)."""
    counts = rng.multinomial(N, np.full(N, 1 / N), size=B).T / N            # [N, B]
    m = X @ counts                                                           # [reps, B] bootstrap means
    lo, hi = np.percentile(m, 2.5, 1), np.percentile(m, 97.5, 1)
    return float(np.mean((lo <= target) & (hi >= target))), float(np.mean(hi - lo))


def cover_exact(X, target):
    out = [intervals.binary_ci(x)[1:] for x in X]
    return float(np.mean([a <= target <= b for a, b in out])), float(np.mean([b - a for a, b in out]))


rows = []


def run(name, pop, design, methods, reps=REPS):
    pop = np.asarray(pop, float)
    X = draw(pop, design, reps)
    for m in methods:
        if m == 'median':
            cov, w = cover_median(X, pop)
        elif m == 'bootstrap':
            cov, w = cover_bootstrap(X, float(pop.mean()))
        else:
            cov, w = cover_exact(X, float(pop.mean()))
        rows.append({'population': name, 'design': design, 'method': m, 'coverage': cov, 'width': w, 'n': N,
                     'reps': reps})
        print(f'{name:38s} {design:6s} {m:9s} coverage {cov:.3f} width {w:.3f}', flush=True)


E = json.load(open(os.path.join(ROOT, 'results', 'eval_r4.json')))
for s, S in E['sets'].items():
    if s == 'controls':
        continue
    for arm, get in (('trained', lambda it, k: it[k]), ('original', lambda it, k: it['orig'][k])):
        for k in ('frac', 'r'):
            pop = [get(it, k) for it in S['items']]
            for design in ('same', 'strata'):
                run(f'observed {s} {arm} {k}', pop, design, ('bootstrap', 'median'))

# synthetic stress populations: large samples stand for the distributions
M = 240_000                                         # divisible by the 12 strata
synthetic = {
    'heavy tail (Student t, 1.5 df)': rng.standard_t(1.5, M),
    'skewed (lognormal, sigma 1.5)': rng.lognormal(0, 1.5, M),
    'outliers (5% far from the rest)': np.where(rng.random(M) < 0.05, rng.normal(5, 3, M), rng.normal(0.7, 0.1, M)),
    'bounded skewed in [-1, 1] (2 Beta(0.3, 3) - 1)': 2 * rng.beta(0.3, 3, M) - 1,
}
for name, pop in synthetic.items():
    for design in ('same', 'strata'):
        run(name, pop, design, ('bootstrap', 'median'))
run('8 of 96 skewed (lognormal, sigma 1.5), the rest narrow', np.concatenate([rng.normal(0.7, 0.1, M // 12 * 11),
                                                                            rng.lognormal(0, 1.5, M // 12)]),
    'groups', ('bootstrap', 'median'))
for p in (0.99, 0.9, 0.5, 0.05):
    pop = (rng.random(M) < p).astype(float)
    for design in ('same', 'strata'):
        run(f'0/1 score, mean {p}', pop, design, ('bootstrap', 'exact'))

json.dump({'alpha': intervals.ALPHA, 'n': N, 'order_statistics': [K1 + 1, K2 + 1], 'rows': rows},
          open(os.path.join(ROOT, 'results', 'interval_coverage.json'), 'w'), indent=1)
