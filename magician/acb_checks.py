"""Checks on the Layer Targeting result of the Activation Controllability Benchmark (Kowalski et al. 2026).

Run from the benchmark's root (github.com/mkobalski/activation-control) with each model's runs under
results/raw/<model>.main and results/raw/<model>.lt, and the magician repo's magician/ dir as argv[1]:

    python acb_checks.py MAGICIAN_DIR OUT.json ORIGINAL MODEL...

For every MODEL against ORIGINAL (the same prompts, paired by sentence and concept):
  (a) each measure's per-unit statistic, from the benchmark's own unit builders: Engage and Suppress, the unit's
      standardized change at the model's peak read layer ((x - b) / sigma_c, whose mean over units is the d');
      Dial Rank, the unit's rank correlation at the peak layer; Temporal Control, the unit's standardized
      contrast averaged over the three temporal conditions; Layer Targeting, the unit's standardized diagonal
      minus the column mean averaged over the 8 instructed layers. For each, the paired difference, an exact
      sign-flip permutation test over the 10 concepts (all 1024 flips) and a Monte Carlo one over the 50
      sentences, and exact order-statistic intervals (intervals.median_ci) on the per-concept and per-sentence
      median differences;
  (b) the per-target profiles over all 20 read layers, raw and column-demeaned, with each target's peak;
  (d) the transcription: compliance and exact-match rates of the layer-targeting prompts.
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "scripts")
import scalar_ci as sc  # noqa: E402
import score_data as sd  # noqa: E402
from score_data import NEG, POS  # noqa: E402

sys.path.insert(0, sys.argv[1])
import intervals  # noqa: E402

COND = "think_at_layer"


def profiles(lt_run, main_run):
    """{(sentence, concept): {T: {L: Δproj[T, L] / σ(c, L)}}} over every read layer L, with the benchmark's
    baseline (the main run's no-instruction cache) and σ (its across-sentence SD per concept and layer)."""
    meta = sd.run_meta(lt_run)
    targets = [int(t) for t in meta["prompt_layers"]]
    reads = [int(L) for L in meta["analysis_layers"]]
    cache = sd.load_baseline(main_run)
    vecs = sd.load_vectors("results/vector_cache", sd._resolve_model(main_run, None), reads, "baseline")
    concepts = vecs[reads[0]][0]
    base = {}
    for s, ent in cache.items():
        n = len(ent["anchored_token_strs"][1:])
        b = {}
        for L in reads:
            A = np.asarray(ent["activations"][L], np.float32)[:n]
            cos = (A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)) @ vecs[L][2].T
            b[L] = (cos * np.asarray(ent["norms"][L], np.float32)[:n, None]).mean(0)
        base[s] = b
    sig = {(c, L): np.std([base[s][L][ci] for s in base], ddof=1) for ci, c in enumerate(concepts) for L in reads}
    out = defaultdict(dict)
    for r in sd.load_rows(lt_run):
        if not (r.get("is_compliant") and r["condition_id"] == COND and r.get("concept")):
            continue
        s, c, T = r["sentence"], r["concept"], int(r["prompt_layer"])
        if s not in base or c not in concepts:
            continue
        ci = concepts.index(c)
        d = {}
        for L in reads:
            cv, nv = sd.trace(r, "cosine_sim", L), sd.trace(r, "norms", L)
            m = min(len(cv), len(nv))
            d[L] = (float(np.nanmean(cv[:m] * nv[:m])) - float(base[s][L][ci])) / sig[(c, L)]
        out[(s, c)][T] = d
    return {k: v for k, v in out.items() if len(v) == len(targets)}, targets, reads


def units(main_run, lt_run, ch="proj"):
    """{measure: {(sentence, concept): per-unit statistic}} for the measures of (a)."""
    out = {}
    layers_list, valsES, basesES = sd.unit_layer_readouts(main_run, [POS, NEG])
    order = sorted(basesES["cos"])
    for cond, sign, name in ((POS, 1.0, "engage"), (NEG, -1.0, "suppress")):
        blocks, sents, kept = sc._blocks_with_concepts(valsES[(ch, cond)], basesES[ch], order)
        z = {}
        for (X, B, idx), c in zip(blocks, kept):
            sd_ = np.nanstd(B, axis=0, ddof=1)
            for row, i in enumerate(idx):
                z[(sents[i], c)] = sign * (X[row] - B[row]) / sd_
        L = int(np.nanargmax(np.nanmean(np.array(list(z.values())), axis=0)))
        out[name] = {k: float(v[L]) for k, v in z.items() if np.isfinite(v[L])}
    UR, _, sid, cid = sc._dial_units(main_run)[ch]
    li = int(np.nanargmax(np.nanmean(UR, axis=0)))
    out["dial_rank"] = {(s, c): float(v) for s, c, v in zip(sid, cid, UR[:, li]) if np.isfinite(v)}
    tb = sc._temporal_blocks(main_run)[ch]
    acc = defaultdict(list)
    for cond in sc.TEMP_CONDS:
        su, sids, cids = tb[cond]
        for v, s, c in zip(su, sids, cids):
            acc[(s, c)].append(float(v))
    out["temporal_control"] = {k: float(np.mean(v)) for k, v in acc.items()}
    D, sid, cid, _ = sd.layer_targeting_units(lt_run, main_run)[ch]
    out["layer_targeting"] = {(s, c): float(np.nanmean(D[i])) for i, (s, c) in enumerate(zip(sid, cid))}
    return out


def paired(a, b, rng):
    """Sign-flip tests and exact intervals for the paired differences a - b over their common units."""
    keys = sorted(set(a) & set(b))
    by_c, by_s = defaultdict(list), defaultdict(list)
    for s, c in keys:
        by_c[c].append(a[(s, c)] - b[(s, c)])
        by_s[s].append(a[(s, c)] - b[(s, c)])
    dc = {c: float(np.mean(v)) for c, v in by_c.items()}
    ds = [float(np.mean(v)) for v in by_s.values()]
    pc, kc, nc = sign_flip(list(dc.values()), rng)
    ps, ks, ns = sign_flip(ds, rng)
    return dict(n_units=len(keys), mean_diff=float(np.mean([a[k] - b[k] for k in keys])),
                concepts=dict(n=len(dc), diffs=dc, n_positive=int(sum(v > 0 for v in dc.values())), p_sign_flip=pc,
                              test=kc, flips=nc, median_ci=list(intervals.median_ci(list(dc.values())))),
                sentences=dict(n=len(ds), n_positive=int(np.sum(np.array(ds) > 0)), p_sign_flip=ps, test=ks,
                               flips=ns, median_ci=list(intervals.median_ci(ds))))


def grid(P, targets, reads):
    """Mean over units of the profiles: (targets x reads), raw and column-demeaned."""
    G = np.array([[np.mean([P[k][T][L] for k in P]) for L in reads] for T in targets])
    return G, G - G.mean(0, keepdims=True)


def transcription(lt_run):
    rows = [r for r in sd.load_rows(lt_run) if r["condition_id"] == COND]
    exact = [r["generated_text"].strip() == r["sentence"].strip() for r in rows]
    return dict(n=len(rows), compliant=float(np.mean([r["is_compliant"] for r in rows])),
                exact=float(np.mean(exact)))


def sign_flip(cluster_diffs, rng, n_mc=200_000):
    """Two-sided p for mean(cluster_diffs) = 0 under sign flips of whole clusters: exact when 2^n <= 2^16."""
    x = np.asarray(cluster_diffs, float)
    obs = abs(x.mean())
    n = len(x)
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        return float(np.mean(np.abs((signs * x).mean(1)) >= obs - 1e-12)), "exact", 2 ** n
    signs = rng.choice((-1.0, 1.0), size=(n_mc, n))
    return float((1 + np.sum(np.abs((signs * x).mean(1)) >= obs - 1e-12)) / (n_mc + 1)), "monte carlo", n_mc


def main():
    out_path, orig, models = sys.argv[2], sys.argv[3], sys.argv[4:]
    rng = np.random.default_rng(0)
    res, U = {}, {}
    for m in [orig] + models:
        lt, mn = f"results/raw/{m}.lt", f"results/raw/{m}.main"
        U[m] = units(mn, lt)
        P, targets, reads = profiles(lt, mn)
        G, Gd = grid(P, targets, reads)
        res[m] = dict(
            unit_means={k: float(np.mean(list(v.values()))) for k, v in U[m].items()},
            n_units={k: len(v) for k, v in U[m].items()},
            targets=targets, reads=reads, grid_raw=G.round(4).tolist(), grid_demeaned=Gd.round(4).tolist(),
            peak_raw={T: reads[int(np.argmax(G[i]))] for i, T in enumerate(targets)},
            peak_demeaned={T: reads[int(np.argmax(Gd[i]))] for i, T in enumerate(targets)},
            # per read layer that is also a target: which instructed layer raises it most (column-demeaned)
            best_target_per_read={L: targets[int(np.argmax(Gd[:, reads.index(L)]))] for L in targets if L in reads},
            transcription=transcription(lt))
        sd.load_rows.cache_clear()
        print(m, {k: round(v, 3) for k, v in res[m]["unit_means"].items()}, "peaks", res[m]["peak_demeaned"],
              flush=True)
    for m in models:
        res[m]["paired_vs_" + orig] = {k: paired(U[m][k], U[orig][k], rng) for k in U[m]}
        for k, v in res[m]["paired_vs_" + orig].items():
            print(m, k, "diff %+.3f" % v["mean_diff"], "concepts %d/%d p=%.4f" % (
                v["concepts"]["n_positive"], v["concepts"]["n"], v["concepts"]["p_sign_flip"]),
                "sentences %d/%d p=%.2g" % (v["sentences"]["n_positive"], v["sentences"]["n"],
                                            v["sentences"]["p_sign_flip"]),
                "median per concept %+.3f [%+.3f, %+.3f]" % tuple(v["concepts"]["median_ci"]), flush=True)
    json.dump(res, open(out_path, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
