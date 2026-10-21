"""Confidence intervals for the per-instruction scores, valid in finite samples without distributional assumptions.

Both intervals assume only that the instructions of a test set are scored independently. They need not be identically
distributed: spec.eval_set gives each instruction a fixed question and draws its concept, wording, and pattern at
random, so every interval is about instructions drawn by this generator from these test pools (questions, concepts,
patterns), not about other pools. Both rest on Hoeffding (1956, Theorem 4): for independent 0/1 variables with mean
success probability p, the number of successes S satisfies P(S <= b) <= P(B <= b) for b <= np - 1 and
P(S >= c) <= P(B >= c) for c >= np, with B binomial(n, p). So a binomial test whose rejection regions sit on those
sides of np keeps its level when the items differ.

median_ci(x)  every continuous score (share of the requested change, pattern match r, and the rest). The median with
    the exact order-statistic interval (binomial; scipy.stats.quantile_test). Its target is the median of the items'
    average distribution function; S is the count of items below it, with p = 1/2, ties included. The ranks of the
    order statistics are checked against Hoeffding's condition below. No interval for a mean of an unbounded score can
    be valid for every distribution (Bahadur and Savage 1956).

binary_ci(x)  a 0/1 score (plan_exact), whose median says little. The mean with the Clopper-Pearson interval, widened
    where Hoeffding's condition fails: the upper end is at least (S + 1) / n, which only changes S = n - 1 and a few
    counts near it.

summary(items, key) returns [estimate, lower, upper, statistic, n]. paired(a, b, key) is the same for the
per-instruction difference a - b of two arms scored on the same instructions.
"""
import numpy as np
from scipy.stats import beta, quantile_test

ALPHA = 0.05
BINARY = ('plan_exact',)                            # 0/1 scores: mean with binary_ci


def median_ci(x, alpha=ALPHA):
    """Median of x and its exact order-statistic interval."""
    x = np.asarray(x, float)
    n = len(x)
    ci = quantile_test(x, p=0.5).confidence_interval(confidence_level=1 - alpha)
    xs = np.sort(x)
    if np.isfinite(ci.low):                                       # 1-based ranks of the order statistics used
        k1 = int(np.searchsorted(xs, ci.low, 'left')) + 1
        assert k1 <= n / 2, (k1, n)                               # Hoeffding's condition for the lower tail
    if np.isfinite(ci.high):
        k2 = int(np.searchsorted(xs, ci.high, 'right'))
        assert k2 >= n / 2 + 1, (k2, n)                           # and for the upper tail
    return float(np.median(x)), float(ci.low), float(ci.high)


def binary_ci(x, alpha=ALPHA):
    """Mean of the 0/1 scores x and its exact interval. The lower end rejects p through S >= s, which needs s >= np:
    the Clopper-Pearson lower end is at most s / n, so it holds. The upper end rejects p through S <= s, which needs
    s <= np - 1, so it is never below (s + 1) / n."""
    x = np.asarray(x, float)
    assert np.isin(x, (0.0, 1.0)).all(), np.unique(x)
    n, s = len(x), int(x.sum())
    lo = beta.ppf(alpha / 2, s, n - s + 1) if s > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, s + 1, n - s) if s < n else 1.0
    assert lo <= s / n <= hi, (s, n, lo, hi)
    return s / n, float(lo), float(max(hi, min(1.0, (s + 1) / n)))


def interval(xs, key, alpha=ALPHA):
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    if len(xs) == 0:
        return None
    if key in BINARY:
        return [*binary_ci(xs, alpha), 'mean', len(xs)]
    return [*median_ci(xs, alpha), 'median', len(xs)]


def summary(items, key, sub=None, alpha=ALPHA):
    """[estimate, lower, upper, statistic, n] of score key over items (sub: the scores of another arm in each item)."""
    return interval([(it[sub] if sub else it).get(key, np.nan) for it in items], key, alpha)


def paired(a, b, key, alpha=ALPHA):
    """[estimate, lower, upper, statistic, n] of the per-instruction difference a - b of a continuous score in two arms
    scored on the same instructions, in the same order: the median difference and its exact interval."""
    assert key not in BINARY, key
    d = np.array([x - y for x, y in zip(a, b)], float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return None
    return [*median_ci(d, alpha), 'median', len(d)]
