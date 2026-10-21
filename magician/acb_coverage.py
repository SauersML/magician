"""Coverage of the Activation Controllability Benchmark's two-way percentile bootstrap (scripts/scalar_ci.py), by
simulation with the observed units as a known population.

Run from the benchmark's root (github.com/mkobalski/activation-control) with a model's runs under
results/raw/<model>.main and results/raw/<model>.lt:

    python acb_coverage.py MODEL OUT.json [N_OUTER] [N_INNER]

The population is the run's own sentences and concepts, and the true value of each measure (and of S) is its value on
the whole population. Each outer draw samples as many sentences and as many concepts as the population has, with
replacement, and is one simulated benchmark run; the benchmark's interval is then the 2.5/97.5 percentiles of N_INNER
two-way resamples of that run (sentences and concepts resampled independently, as scalar_ci.py does), and coverage is
the share of outer draws whose interval holds the true value, with its exact binomial interval (intervals.binary_ci)
over the N_OUTER draws. All draws are multiplicity weights on the population, so
every statistic goes through the benchmark's own kernels: Engage and Suppress are the maximum over read layers of a d',
Coverage the minimum over word classes, Dial Rank, Temporal Control and Layer Targeting means of unit contrasts.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
import compute_scores as cs  # noqa: E402
import scalar_ci as sc  # noqa: E402
import score_data as sd  # noqa: E402
from score_data import NEG, POS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intervals  # noqa: E402

CH = "proj"


def prepare(run, lt_run):
    """The benchmark's per-measure inputs, and evaluate(Wsent, Mconc) -> {measure: (B,) replicate values, 'S': (B,)}
    for sentence and concept multiplicity matrices over the population (B x n_sentences, B x n_concepts)."""
    layers_list, valsES, basesES = sd.unit_layer_readouts(run, [POS, NEG])
    order_es = sorted(basesES["cos"])
    valsCov, basesCov = sd.pos_category_readouts(run, [POS])
    dial = sc._dial_units(run)
    temporal = sc._temporal_blocks(run)
    layer = sc._layer_targeting_block(lt_run, run)
    es = {cond: sc._blocks_with_concepts(valsES[(CH, cond)], basesES[CH], order_es) for cond in (POS, NEG)}
    cov = sc._blocks_with_concepts(valsCov[(CH, POS)], basesCov[CH], sorted(basesCov[CH]))
    S_all, C_all = set(), set()
    for _, sents, kept in list(es.values()) + [cov]:
        S_all.update(sents), C_all.update(kept)
    UR, _, sid, cid = dial[CH]
    S_all.update(sid), C_all.update(cid)
    for cond in sc.TEMP_CONDS:
        _, sids, cids = temporal[CH][cond]
        S_all.update(sids), C_all.update(cids)
    D, lsid, lcid = layer[CH]
    S_all.update(lsid), C_all.update(lcid)
    S_all, C_all = sorted(S_all), sorted(C_all)
    si, ci = {s: i for i, s in enumerate(S_all)}, {c: i for i, c in enumerate(C_all)}
    li = int(np.nanargmax(np.nanmean(UR, axis=0)))
    dummy = np.random.default_rng(0)

    def evaluate(Wsent, Mconc):
        pw = lambda ss: Wsent[:, [si[s] for s in ss]]
        pm = lambda cc: Mconc[:, [ci[c] for c in cc]]
        out = {}
        for cond, sign, name in ((POS, 1.0, "engage"), (NEG, -1.0, "suppress")):
            blocks, sents, kept = es[cond]
            st = cs.dprime_stats(blocks, len(sents), len(layers_list), dummy, W=pw(sents), Mc=pm(kept), n_perm=0)
            out[name] = np.nanmax(sign * st["bavg"], axis=1)
        cb, cs_, ck = cov
        st = cs.dprime_stats(cb, len(cs_), len(sd.CATS), dummy, W=pw(cs_), Mc=pm(ck), n_perm=0)
        out["coverage"] = np.nanmin(st["bavg"], axis=1)
        sa, ca = cs.twoway_axes([(UR, sid, cid)])
        _, out["dial_rank"] = cs.twoway([(UR[:, li:li + 1], sid, cid)], stat=sc._rank_stat, Ws=pw(sa), Wc=pm(ca),
                                        return_reps=True)
        blocks = [temporal[CH][c] for c in sc.TEMP_CONDS if len(temporal[CH][c][0])]
        sa, ca = cs.twoway_axes(blocks)
        _, out["temporal_control"] = cs.twoway(blocks, stat=sc._temp_stat, Ws=pw(sa), Wc=pm(ca), return_reps=True)
        sa, ca = cs.twoway_axes([(D, lsid, lcid)])
        _, out["layer_targeting"] = cs.twoway([(D, lsid, lcid)], stat=sc._layer_stat, Ws=pw(sa), Wc=pm(ca),
                                              return_reps=True)
        logG = sum(np.log(sc._p_rank(v) if m == "dial_rank" else sc._p_score(m, v)) for m, v in out.items()) / len(out)
        out["S"] = np.clip(2.0 * np.exp(logG) - 1.0, 0.0, 1.0)
        return {k: np.asarray(v, float) for k, v in out.items()}

    return evaluate, len(S_all), len(C_all), li


def main():
    model = sys.argv[1]
    out_path = sys.argv[2]
    n_outer = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    n_inner = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    evaluate, nS, nC, li = prepare(f"results/raw/{model}.main", f"results/raw/{model}.lt")
    truth = {k: float(v[0]) for k, v in evaluate(np.ones((1, nS)), np.ones((1, nC))).items()}
    print(model, "population", nS, "sentences", nC, "concepts; truth", {k: round(v, 3) for k, v in truth.items()},
          flush=True)
    rng = np.random.default_rng(1)
    hits = {k: [] for k in truth}
    width = {k: [] for k in truth}
    t0 = time.time()
    for r in range(n_outer):
        a = rng.multinomial(nS, np.full(nS, 1 / nS))
        c = rng.multinomial(nC, np.full(nC, 1 / nC))
        Ws = rng.multinomial(nS, a / nS, size=n_inner).astype(float)
        Wc = rng.multinomial(nC, c / nC, size=n_inner).astype(float)
        reps = evaluate(Ws, Wc)
        for k, v in reps.items():
            v = v[np.isfinite(v)]
            lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
            hits[k].append(bool(lo <= truth[k] <= hi))
            width[k].append(float(hi - lo))
        if r % 10 == 9:
            print(r + 1, f"{time.time() - t0:.0f}s", {k: round(float(np.mean(h)), 3) for k, h in hits.items()},
                  flush=True)
    res = dict(model=model, n_outer=n_outer, n_inner=n_inner, n_sentences=nS, n_concepts=nC, truth=truth,
               coverage={k: float(np.mean(h)) for k, h in hits.items()},
               coverage_ci={k: list(intervals.binary_ci(h)[1:]) for k, h in hits.items()},
               hits={k: int(np.sum(h)) for k, h in hits.items()},
               median_width={k: float(np.median(w)) for k, w in width.items()})
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res["coverage"]))


if __name__ == "__main__":
    main()
