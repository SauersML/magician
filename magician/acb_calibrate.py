"""Calibrated 95% intervals for the Activation Controllability Benchmark's S and its six measures: the benchmark's own
two-way bootstrap (scripts/scalar_ci.py: sentences and concepts resampled independently), with each tail's level set
by a double bootstrap (Hall's calibration, one tail at a time).

Run from the benchmark's root (github.com/mkobalski/activation-control) with a model's runs under
results/raw/<model>.main and results/raw/<model>.lt:

    python acb_calibrate.py MODEL OUT.json [N_OUTER] [N_INNER] [N_BOOT]

The population is the run's own sentences and concepts, and the true value of each measure is its value on the whole
population. An outer draw resamples as many sentences and concepts as the population has, with replacement (one
simulated benchmark run), and its N_INNER two-way resamples give two intervals at every tail level q in Q: the
percentile interval [Q(q), Q(1 - q)] and the basic interval [2t - Q(1 - q), 2t - Q(q)], where Q is the inner quantile
and t the outer draw's own value (the basic interval corrects a biased statistic, such as a maximum over layers). For
each measure, kind of interval, and tail, the calibrated level is the largest q at which at most 2.5% of the N_OUTER
outer draws put the true value beyond that tail. The kind used is the one reaching 2.5% on both tails with the
narrower median width over the same draws; if neither reaches it, the one missing least at the widest level, and the
measure is flagged under_covers. A fresh, independent set of N_OUTER outer draws gives the coverage of the chosen
interval, with its exact binomial interval (intervals.binary_ci); a measure whose coverage interval lies below 0.95 is
also flagged. The reported interval is the observed run's N_BOOT two-way resamples at the calibrated levels; the
point value is the benchmark's. Every statistic goes through the benchmark's own kernels (acb_coverage.prepare).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acb_coverage  # noqa: E402
import intervals  # noqa: E402

Q = np.array([0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.03, 0.035,
              0.04, 0.05, 0.06, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25])
TAIL = 0.025
KINDS = ("percentile", "basic")


def ends(kind, t, v):
    """(lower ends at levels Q, upper ends at levels Q) of one kind of interval from replicates v around value t."""
    lo, hi = np.quantile(v, Q), np.quantile(v, 1 - Q)
    return (lo, hi) if kind == "percentile" else (2 * t - hi, 2 * t - lo)


def draws(evaluate, nS, nC, keys, n_outer, n_inner, seed, label):
    """{kind: {measure: (lower, upper)}}, each (n_outer, len(Q)): both intervals' ends at every level, per outer
    draw. Row 0 of each evaluation is the outer draw itself, the rest its inner resamples."""
    rng = np.random.default_rng(seed)
    out = {kind: {k: (np.full((n_outer, len(Q)), np.nan), np.full((n_outer, len(Q)), np.nan)) for k in keys}
           for kind in KINDS}
    t0 = time.time()
    for r in range(n_outer):
        a = rng.multinomial(nS, np.full(nS, 1 / nS))
        c = rng.multinomial(nC, np.full(nC, 1 / nC))
        Ws = np.vstack([a, rng.multinomial(nS, a / nS, size=n_inner)]).astype(float)
        Wc = np.vstack([c, rng.multinomial(nC, c / nC, size=n_inner)]).astype(float)
        for k, v in evaluate(Ws, Wc).items():
            t, v = v[0], v[1:][np.isfinite(v[1:])]
            for kind in KINDS:
                lo, hi = ends(kind, t, v)
                out[kind][k][0][r], out[kind][k][1][r] = lo, hi
        if r % 100 == 99:
            print(label, r + 1, f"{time.time() - t0:.0f}s", flush=True)
    return out


def observed(evaluate, nS, nC, n_boot, seed, chunk=1000):
    """{measure: (n_boot,) replicate values} of the observed run's two-way bootstrap."""
    rng = np.random.default_rng(seed)
    out = {}
    for b in range(0, n_boot, chunk):
        m = min(chunk, n_boot - b)
        Ws = rng.multinomial(nS, np.full(nS, 1 / nS), size=m).astype(float)
        Wc = rng.multinomial(nC, np.full(nC, 1 / nC), size=m).astype(float)
        for k, v in evaluate(Ws, Wc).items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def calibrate(lo, hi, truth):
    """(i_lo, i_hi, reached, miss below, miss above): on each tail, the narrowest level such that it and every wider
    level miss at most TAIL of the draws (NaN ends count as misses), or the widest level if that one misses more."""
    below = ~(lo <= truth)
    above = ~(hi >= truth)
    i_lo, i_hi = widest_run(below.mean(0) <= TAIL), widest_run(above.mean(0) <= TAIL)
    ok = i_lo is not None and i_hi is not None
    i_lo, i_hi = i_lo or 0, i_hi or 0
    return i_lo, i_hi, ok, float(below[:, i_lo].mean()), float(above[:, i_hi].mean())


def widest_run(ok):
    """The last index of the run of levels that pass, starting from the widest (index 0); None if the widest fails."""
    if not ok[0]:
        return None
    return int(len(ok) - 1 if ok.all() else np.argmin(ok) - 1)


def main():
    model, out_path = sys.argv[1], sys.argv[2]
    n_outer = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    n_inner = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    n_boot = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
    evaluate, nS, nC, _ = acb_coverage.prepare(f"results/raw/{model}.main", f"results/raw/{model}.lt")
    truth = {k: float(v[0]) for k, v in evaluate(np.ones((1, nS)), np.ones((1, nC))).items()}
    print(model, nS, "sentences", nC, "concepts; values", {k: round(v, 3) for k, v in truth.items()}, flush=True)
    cal = draws(evaluate, nS, nC, truth, n_outer, n_inner, 1, "calibration")
    chk = draws(evaluate, nS, nC, truth, n_outer, n_inner, 2, "check")
    reps = observed(evaluate, nS, nC, n_boot, 3)
    i95 = int(np.flatnonzero(np.isclose(Q, TAIL))[0])
    res = {}
    for k, T in truth.items():
        fits = {}
        for kind in KINDS:
            lo, hi = cal[kind][k]
            i_lo, i_hi, ok, mb, ma = calibrate(lo, hi, T)
            fits[kind] = dict(i_lo=i_lo, i_hi=i_hi, ok=ok, miss=mb + ma, below=mb, above=ma,
                              width=float(np.nanmedian(hi[:, i_hi] - lo[:, i_lo])))
        good = [kd for kd in KINDS if fits[kd]["ok"]]
        kind = min(good, key=lambda kd: fits[kd]["width"]) if good else min(KINDS, key=lambda kd: fits[kd]["miss"])
        f = fits[kind]
        lo, hi = chk[kind][k]
        hit = (lo[:, f["i_lo"]] <= T) & (hi[:, f["i_hi"]] >= T)
        cov, cov_lo, cov_hi = intervals.binary_ci(hit.astype(float))
        plo, phi = chk["percentile"][k]
        v = reps[k][np.isfinite(reps[k])]
        olo, ohi = ends(kind, T, v)
        olo, ohi = float(olo[f["i_lo"]]), float(ohi[f["i_hi"]])
        if k == "S":
            olo, ohi = max(olo, 0.0), min(ohi, 1.0)
        res[k] = dict(obs=T, lo=olo, hi=ohi, kind=kind, tail_lo=float(Q[f["i_lo"]]), tail_hi=float(Q[f["i_hi"]]),
                      calibration_miss_below=f["below"], calibration_miss_above=f["above"],
                      coverage=cov, coverage_ci=[cov_lo, cov_hi],
                      coverage_uncalibrated=float(((plo[:, i95] <= T) & (phi[:, i95] >= T)).mean()),
                      under_covers=bool(not f["ok"] or cov_hi < 0.95),
                      calibration={kd: {a: b for a, b in fits[kd].items()} for kd in KINDS},
                      n_boot_finite=int(len(v)))
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[k].items() if a != "calibration"},
              flush=True)
    S = res.pop("S")
    out = dict(model=model, method="two-way bootstrap, double-bootstrap calibrated to 95%",
               scalar=S["obs"], ci_lo=S["lo"], ci_hi=S["hi"], S=S,
               components={f"{k}|proj": v for k, v in res.items()},
               coverage={**{k: v["coverage"] for k, v in res.items()}, "S": S["coverage"]},
               under_covers={**{k: v["under_covers"] for k, v in res.items()}, "S": S["under_covers"]},
               n_outer_calibration=n_outer, n_outer_check=n_outer, n_inner=n_inner, n_boot=n_boot,
               tail_levels=Q.tolist(), n_sentences=nS, n_concepts=nC)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
