"""Cartilage populations, historical symbols and independent anatomical controls."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
from collections import Counter
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from scipy.stats import spearmanr, binomtest
from sklearn.cluster import KMeans
from ghir_methods import (
    EPS,
    MARKER_MODULES,
    CANONICAL_GH_IGF,
    gene_lookup,
    select_hvg_and_pca,
    annotate_states,
    graph_geodesic_pseudotime,
    qc_and_normalize_sc,
    fit_response_window,
    smooth_binned_profile,
    weighted_gh_score,
    _sparse_gene_moments,
    _rank_quantile_bins,
    response_window_sensitivity,
    signature_weighting_sensitivity,
    pseudotime_root_knn_robustness,
    parse_geo_family_soft,
    platform_probe_to_gene,
    extract_value_column,
    clean_gene_symbol,
    percentile_score,
    construct_gh_weights,
    export_panel,
)
import matplotlib.pyplot as plt

LINEAGES = {
    "cartilage": ["Col2a1", "Acan", "Sox9", "Comp", "Col9a1"],
    "immune": ["Ptprc", "Lyz2", "Tyrobp"],
    "endothelial": ["Pecam1", "Cdh5", "Kdr"],
    "bone/perichondrium": ["Col1a1", "Col1a2", "Sp7", "Bglap"],
    "interzone": ["Gdf5", "Wnt9a", "Prg4"],
    "epithelial": ["Epcam", "Krt8", "Krt18"],
}
SEX_PANEL = ["Xist", "Ddx3y", "Kdm5d", "Eif2s3y", "Uty"]
DATA_P28 = "GSM8769462_3402_filtered_feature_bc_matrix.h5"


def read_postnatal(inputs):
    """Recover sparse gene-expression counts from the deposited 10x HDF5 file."""
    path = inputs / DATA_P28
    # Python opens the Unicode path. h5py receives a binary stream and closes first.
    with path.open("rb") as source, h5py.File(source, "r") as f:
        m = f["matrix"]
        X = sparse.csc_matrix(
            (m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])
        ).T.tocsr()
        features = m["features/name"].asstr()[:]
        use = m["features/feature_type"].asstr()[:] == "Gene Expression"
        bc = m["barcodes"].asstr()[:]
    return qc_and_normalize_sc(X[:, use], features[use], bc, max_cells=None)


def subset_lineages(X, genes, meta, dataset, out):
    """Choose clusters before scoring the exposure signature, with the same rule at both ages."""
    pcs, _, _ = select_hvg_and_pca(X, genes)
    labels = KMeans(n_clusters=18, n_init=10, random_state=42).fit_predict(pcs)
    lookup = gene_lookup(genes)
    scores = pd.DataFrame(
        {
            k: np.asarray(
                X[:, [lookup[g.upper()] for g in gs if g.upper() in lookup]].mean(axis=1)
            ).ravel()
            for k, gs in LINEAGES.items()
        }
    )
    profile = scores.groupby(labels).mean()
    profile["assignment"] = profile.idxmax(axis=1)
    profile["n"] = np.bincount(labels)
    lineage = profile.loc[labels, "assignment"].to_numpy()
    keep = lineage == "cartilage"
    profile.rename_axis("cluster").to_csv(out / f"{dataset}_lineages.csv")
    pd.DataFrame(
        {"barcode": meta.barcode, "cluster": labels, "lineage": lineage, "retained": keep}
    ).to_csv(out / f"{dataset}_selection.csv.gz", index=False)
    return keep, lineage


def selected_mappings(gh, mgi, genes):
    """Audit source labels individually and sum signed weights when aliases converge."""
    lookup = gene_lookup(genes)
    official = mgi[mgi.match == "official"].set_index("array_label").official.to_dict()
    aliases = mgi.groupby("array_label").official.agg(lambda x: tuple(sorted(set(x)))).to_dict()
    ids = []
    for gene, w in gh.loc[gh.weight.ne(0), "weight"].items():
        key = str(gene).upper()
        candidates = (official[key],) if key in official else aliases.get(key, ())
        if key in lookup:
            target, rule = key, "direct"
        elif len(candidates) == 1 and candidates[0] in lookup:
            target, rule = candidates[0], "unique_alias"
        else:
            target, rule = None, ("ambiguous" if len(candidates) > 1 else "unmapped")
        ids.append((key, w, rule, target, "|".join(candidates)))
    return pd.DataFrame(ids, columns=["source", "weight", "rule", "target", "mgi_candidates"])


def training_controls(zmeans, gh, alias_table):
    """Select zone-specific controls using only newborn array means and fixed marker exclusions."""
    excluded = {
        g.upper()
        for panel in list(LINEAGES.values()) + list(MARKER_MODULES.values())
        for g in panel
    }
    excluded.update(g.upper() for g in SEX_PANEL)
    excluded.update(gh.index[gh.weight.ne(0)])
    excluded.update(
        alias_table.loc[alias_table.array_label.isin(gh.index[gh.weight.ne(0)]), "official"]
    )
    data = zmeans.drop(index=list(excluded), errors="ignore").dropna()
    frames = []
    for zone in ("resting", "hypertrophic"):
        competitors = data.drop(columns=zone).max(axis=1)
        ratio = np.log2((data[zone] + 1) / (competitors + 1))
        tab = pd.DataFrame({"score": ratio, "source_intensity": data[zone]})
        top = (
            tab.loc[(tab.source_intensity >= 100) & (tab.score > 0)]
            .sort_index()
            .sort_values("score", ascending=False, kind="stable")
            .head(50)
            .copy()
        )
        assert len(top) == 50
        top["signature"] = zone
        frames.append(top.rename_axis("gene").reset_index())
    controls = pd.concat(frames, ignore_index=True)
    assert controls.gene.is_unique
    return controls


@dataclass
class Population:
    name: str
    X: object
    genes: list
    meta: object
    pcs: object
    grid: object
    centers: object
    z_bins: object
    mean: object
    sd: object
    detection: object
    strata: object
    eligible: object
    central: object
    early: object
    late: object


def prepare_population(name, X, genes, meta, exclude_graph=()):
    """Build an independent graph and cache sufficient statistics for gene-set curves."""
    if exclude_graph:
        mask = np.array([g.upper() not in exclude_graph for g in genes])
        pcs, _, _ = select_hvg_and_pca(X[:, mask], list(np.asarray(genes)[mask]))
    else:
        pcs, _, _ = select_hvg_and_pca(X, genes)
    scores, _ = annotate_states(X, genes)
    pt, _, _ = graph_geodesic_pseudotime(pcs, scores)
    meta = (
        meta[["barcode", "total_counts", "n_genes", "pct_mt"]].reset_index(drop=True).join(scores)
    )
    meta["pseudotime"] = pt
    meta["PC1"], meta["PC2"] = pcs[:, 0], pcs[:, 1]
    mean, sd, det = _sparse_gene_moments(X)
    ok = np.isfinite(pt)
    ix = np.flatnonzero(ok)
    bins = np.clip(np.digitize(pt[ok], np.linspace(0, 1, 31)) - 1, 0, 29)
    n = np.bincount(bins, minlength=30)
    avg = sparse.csr_matrix((1 / n[bins], (bins, ix)), shape=(30, len(pt)))
    used = n > 0
    z = ((avg @ X).toarray()[used] - mean) / np.maximum(sd, EPS)
    centers = (np.flatnonzero(used) + 0.5) / 30
    grid = np.linspace(centers.min(), centers.max(), 300)
    q10, q30, q70, q90 = np.quantile(pt[ok], [0.1, 0.3, 0.7, 0.9])
    eligible = (sd > EPS) & (det > 0)
    variability = np.sqrt(
        np.average((z - np.average(z, axis=0, weights=n[used])) ** 2, axis=0, weights=n[used])
    )
    strata = np.zeros(len(genes), int)
    for j, v in enumerate((mean, det, variability)):
        strata[eligible] += 4**j * _rank_quantile_bins(v[eligible], 4)
    return Population(
        name,
        X,
        list(genes),
        meta,
        pcs,
        grid,
        centers,
        z,
        mean,
        sd,
        det,
        strata,
        eligible,
        (grid >= q10) & (grid <= q90),
        (grid >= q10) & (grid <= q30),
        (grid >= q70) & (grid <= q90),
    )


def standardized_curve(pop, idx, weights):
    b = pop.z_bins[:, idx] @ weights / np.linalg.norm(weights)
    return smooth_binned_profile(pop.centers, b, pop.grid)


def high_interval(grid, curve):
    peak = int(np.argmax(curve))
    above = curve >= curve.min() + 0.8 * np.ptp(curve)
    start = end = peak
    while start > 0 and above[start - 1]:
        start -= 1
    while end + 1 < len(curve) and above[end + 1]:
        end += 1
    return grid[peak], grid[start], grid[end]


def signature_test(pop, weights, label, seed, out, direction=0, n_draws=1000):
    """Matched random gene sets and optional weight permutations on a fixed graph.

    The sign of the directional contrast is specified by the source zone. For GH
    only nondirectional amplitudes are interpreted. Calibration draws are independent
    of the matched negative-control draws used to set the reference distribution.
    """
    lookup = gene_lookup(pop.genes)
    mapped = [g for g in weights.index if g in lookup and pop.eligible[lookup[g]]]
    idx = np.array([lookup[g] for g in mapped])
    w = weights.loc[mapped].to_numpy(float)
    assert len(idx) > 4 and len(np.unique(idx)) == len(idx)
    control = pop.eligible.copy()
    control[idx] = False
    pools = [
        (np.flatnonzero(control & (pop.strata == s)), int((pop.strata[idx] == s).sum()))
        for s in np.unique(pop.strata[idx])
    ]
    assert all(len(pool) >= n for pool, n in pools)
    metrics = lambda y: np.array(
        [np.ptp(y), np.ptp(y[pop.central]), y[pop.late].mean() - y[pop.early].mean()]
    )
    observed_curve = standardized_curve(pop, idx, w)
    obs = metrics(observed_curve)
    rng = np.random.default_rng(seed)

    def matched():
        sample = np.concatenate([rng.choice(pool, n, replace=False) for pool, n in pools])
        return metrics(standardized_curve(pop, sample, rng.permutation(w)))

    historical_null = pop.name == "original" and label == "GH"
    if historical_null:
        # Preserve the submission follow-up's published seed and draw order.
        rng = np.random.default_rng(42017)
        shuffles = []
        draws = []
        for _ in range(n_draws):
            shuffles.append(metrics(standardized_curve(pop, idx, rng.permutation(w))))
            draws.append(matched())
        shuffles = np.vstack(shuffles)
        draws = np.vstack(draws)
    else:
        draws = np.vstack([matched() for _ in range(n_draws)])

    def probability(a, b):
        return (1 + np.count_nonzero(a >= b)) / (1 + len(a))

    peak, lo, hi = high_interval(pop.grid, observed_curve)
    midpeak, midlo, midhi = high_interval(pop.grid[pop.central], observed_curve[pop.central])
    row = dict(
        atlas=pop.name,
        signature=label,
        n_active=len(idx),
        peak=peak,
        lower=lo,
        upper=hi,
        interior_peak=midpeak,
        interior_lower=midlo,
        interior_upper=midhi,
        full_amplitude=obs[0],
        amplitude=obs[1],
        late_minus_early=obs[2],
        full_matched_p=probability(draws[:, 0], obs[0]),
        matched_p=probability(draws[:, 1], obs[1]),
    )
    null = pd.DataFrame(draws, columns=["full", "interior", "late_minus_early"])
    null["null"] = "matched"
    if direction:
        row["direction_p"] = probability(direction * draws[:, 2], direction * obs[2])
        neg = np.vstack([matched() for _ in range(200)])
        row["negative_hits"] = sum(probability(draws[:, 1], a) < 0.05 for a in neg[:, 1])
        nc = pd.DataFrame(neg, columns=["full", "interior", "late_minus_early"])
        nc["null"] = "negative_control"
        null = pd.concat([null, nc])
    else:
        if not historical_null:
            shuffles = np.vstack(
                [metrics(standardized_curve(pop, idx, rng.permutation(w))) for _ in range(n_draws)]
            )
        row["shuffle_p"] = probability(shuffles[:, 1], obs[1])
        ndf = pd.DataFrame(shuffles, columns=["full", "interior", "late_minus_early"])
        ndf["null"] = "weight_shuffle"
        null = pd.concat([null, ndf])
    key = pop.name + "_" + label
    null.to_csv(out / (key + "_draws.csv.gz"), index=False)
    pd.DataFrame(
        {"pseudotime": pop.grid, "fitted": observed_curve, "interior": pop.central}
    ).to_csv(out / (key + "_curve.csv"), index=False)
    return row


def expression_evidence(pop, bounds, out):
    """Summarize measured transcripts and depth-stratified marker evidence without assigning donors."""
    m = pop.meta.copy()
    pt = m.pseudotime.to_numpy()
    inside = (pt >= bounds[0]) & (pt <= bounds[1])
    ix = gene_lookup(pop.genes)
    genes = ["Ghr", "Jak2", "Stat5a", "Stat5b", "Cish", "Socs2", "Igf1", "Igf1r"]
    rows = []
    for name, mask in [("inside", inside), ("outside", np.isfinite(pt) & ~inside)]:
        for g in genes:
            vals = pop.X[mask, ix[g.upper()]]
            rows.append(
                (pop.name, name, g, int(vals.getnnz()), int(mask.sum()), float((vals > 0).mean()))
            )
    pd.DataFrame(rows, columns=["atlas", "group", "gene", "detected", "total", "fraction"]).to_csv(
        out / f"{pop.name}_receptors.csv", index=False
    )
    for g in SEX_PANEL:
        m[g] = pop.X[:, ix[g.upper()]].toarray().ravel() > 0 if g.upper() in ix else False
    y = m[SEX_PANEL[1:]].sum(axis=1)
    m["evidence"] = np.select(
        [m.Xist & y.eq(0), ~m.Xist & y.ge(2), m.Xist & y.gt(0)],
        ["Xist only", "multiple Y", "mixed"],
        default="indeterminate",
    )
    m["interval"] = inside
    m["quartile"] = pd.qcut(pt, 4, labels=False, duplicates="drop")
    m["depth_bin"] = pd.qcut(m.total_counts.rank(method="first"), 10, labels=False)
    m.to_csv(out / f"{pop.name}_cells.csv.gz", index=False)
    tab = pd.crosstab(m.quartile, m.evidence).reindex(
        columns=["Xist only", "multiple Y", "mixed", "indeterminate"], fill_value=0
    )
    tab.to_csv(out / f"{pop.name}_sex_by_quartile.csv")
    m.loc[np.isfinite(pt)].groupby("interval")[
        ["score_gh_transduction", "score_igf_axis"]
    ].mean().to_csv(out / f"{pop.name}_modules.csv")
    supported = m[m.evidence.isin(["Xist only", "multiple Y"]) & m.pseudotime.notna()].reset_index(
        drop=True
    )
    sexrow = dict(atlas=pop.name, **m.evidence.value_counts().to_dict())
    if supported.evidence.nunique() == 2 and supported.evidence.value_counts().min() >= 10:
        t = supported.pseudotime.to_numpy()
        flag = supported.evidence.eq("multiple Y").to_numpy()
        groups = list(supported.groupby("depth_bin").indices.values())
        diff = lambda mask: t[mask].mean() - t[~mask].mean()
        delta = diff(flag)
        rng = np.random.default_rng(1118)
        n = 0
        for _ in range(999):
            b = flag.copy()
            for inds in groups:
                b[inds] = rng.permutation(flag[inds])
            # Conditional null preserves group imbalances across depth strata.
            expected = (
                sum(t[inds].mean() * flag[inds].sum() for inds in groups) / flag.sum()
                - sum(t[inds].mean() * (~flag[inds]).sum() for inds in groups) / (~flag).sum()
            )
            n += abs(diff(b) - expected) >= abs(delta - expected)
        sexrow.update(
            mean_pt_difference=delta,
            depth_expected_difference=expected,
            conditional_p=(1 + n) / 1000,
        )
    return m, sexrow


def array_marker_audit(inputs, gh, mgi, out):
    """Trace official/legacy symbols to platform probes and their eligibility. Inspect zonal sex probes separately."""
    table = parse_geo_family_soft(inputs / "GSE3689_family.soft.gz")["platforms"]["GPL922"]["table"]
    mapping = platform_probe_to_gene(table).str.upper()
    reference = {
        "IGF1": (2.60, 2.29),
        "WNT5A": (1.87, 2.03),
        "BMP2": (0.74, 0.78),
        "BMP7": (None, 0.51),
        "STAT5A": (None, 0.60),
        "TBX3": (2.0, 2.5),
        "IGFBP3": (1.37, 1.54),
        "IGF2BP1": (2.11, 2.55),
        "DVL2": (0.83, 0.81),
        "STAT1": (None, 0.44),
        "STAT4": (None, 0.40),
        "STAT6": (None, 1.42),
        "FBXW4": (1.43, None),
        "SMAD5": (None, 2.77),
    }
    rows = []
    qc = []
    for gene in sorted(CANONICAL_GH_IGF | set(reference)):
        records = mgi.loc[mgi.official.eq(gene)]
        labels = sorted(set(records.array_label) | {gene})
        eligible = []
        for label in labels:
            candidates = set(mgi.loc[mgi.array_label.eq(label), "official"])
            if label != gene and (candidates != {gene}):
                continue
            probes = mapping[mapping.eq(label)]
            if label == gene or len(probes):
                x = gh.loc[label] if label in gh.index else {}
                rows.append(
                    dict(
                        gene=gene,
                        source_label=label,
                        probe_count=len(probes),
                        probe_ids="|".join(probes.index.astype(str)),
                        n6=x.get("n_valid_6h", 0),
                        n24=x.get("n_valid_24h", 0),
                        selected=bool(x.get("selected_signature", False)),
                        rank=x.get("rank_abs_weight", np.nan),
                        weight=x.get("weight", np.nan),
                    )
                )
                if label in gh.index:
                    eligible.append(label)
        if gene in reference:
            for time, fc in zip(["6h", "24h"], reference[gene]):
                if fc is None:
                    continue
                vals = [
                    gh.loc[g, "log2fc_" + time]
                    for g in eligible
                    if gh.loc[g, "n_valid_" + time] >= 2
                ]
                qc.append(
                    dict(
                        gene=gene,
                        time=time,
                        published_log2fc=np.log2(fc),
                        estimated_log2fc=float(np.median(vals)) if vals else np.nan,
                        source_labels="|".join(eligible),
                        published_ratio=fc,
                    )
                )
    pd.DataFrame(rows).to_csv(out / "canonical_probe_audit.csv", index=False)
    pd.DataFrame(qc).to_csv(out / "expanded_target_qc.csv", index=False)
    sex = []
    for acc in ["GSE87605", "GSE14007"]:
        soft = parse_geo_family_soft(inputs / f"{acc}_family.soft.gz")
        probes = platform_probe_to_gene(soft["platforms"]["GPL1261"]["table"]).str.upper()
        for gsm, v in soft["samples"].items():
            df = v["table"].set_index("ID_REF")
            values = pd.to_numeric(df[extract_value_column(df)], errors="coerce")
            for g in SEX_PANEL:
                p = probes[probes.eq(g.upper())].index.intersection(values.index)
                sex.append(
                    dict(
                        accession=acc,
                        sample=gsm,
                        title=v["metadata"]["Sample_title"][0],
                        gene=g,
                        n_probes=len(p),
                        signal=values.loc[p].median() if len(p) else np.nan,
                        relative_to_array_median=values.loc[p].median() / values.median()
                        if len(p)
                        else np.nan,
                    )
                )
    pd.DataFrame(sex).to_csv(out / "zonal_sex_probes.csv", index=False)


def leave_gene_out(pop, gh, out, historical=False):
    """Refit the interval after deleting each gene and measure only its held-out contrast."""
    selected = gh.loc[gh.weight.ne(0)]
    ix = gene_lookup(pop.genes)
    used = [g for g in selected.index if g in ix and pop.eligible[ix[g]]]
    idx = np.array([ix[g] for g in used])
    w = selected.loc[used, "weight"].to_numpy()
    t = pop.meta.pseudotime.to_numpy()
    # Historical estimates use the original full range. In the new population,
    # comparison uses the improved interior threshold because of sparse ends.
    gridmask = np.ones(len(pop.grid), bool) if historical else pop.central
    grid = pop.grid[gridmask]
    z = pop.z_bins[:, idx]
    numerator = z @ w
    denominator_sq = np.dot(w, w)
    if historical:
        mapped_weights = selected.loc[[g for g in selected.index if g in ix], "weight"].to_numpy()
        denominator_sq = np.dot(mapped_weights, mapped_weights)
    dense = pop.X[:, idx].toarray()
    rows = []
    for j, g in enumerate(used):
        y = (numerator - w[j] * z[:, j]) / np.sqrt(denominator_sq - w[j] ** 2)
        curve = smooth_binned_profile(pop.centers, y, grid)
        pk, lo, hi = high_interval(grid, curve)
        a = (t >= lo) & (t <= hi)
        b = np.isfinite(t) & ~a
        x = dense[:, j]
        row = dict(
            gene=g,
            weight=abs(selected.loc[g, "weight_raw"]),
            peak=pk,
            lower=lo,
            upper=hi,
            n_inside=a.sum(),
            n_outside=b.sum(),
            smd=np.nan,
            prevalence=np.nan,
        )
        if min(a.sum(), b.sum()) >= 10:
            v1 = x[a]
            v0 = x[b]
            pooled = np.sqrt(
                ((len(v1) - 1) * v1.var() + (len(v0) - 1) * v0.var()) / (len(v1) + len(v0) - 2)
            )
            row.update(
                smd=(v1.mean() - v0.mean()) / (pooled + EPS),
                prevalence=max(np.mean(v1 > 0), np.mean(v0 > 0)),
            )
        rows.append(row)
    df = pd.DataFrame(rows).set_index("gene")
    if historical:
        ranks = pd.read_csv(out / "ranks.csv")
        fixed = ranks.loc[ranks.weight_method.eq("primary")].set_index("gene").candidate_rank
        eligible = df.index.isin(fixed.index) & df.smd.notna()
        df["original_rank"] = fixed.reindex(df.index)
    else:
        eligible = df.prevalence.ge(0.05) & df.smd.notna()
    raw = df.loc[eligible, "weight"]
    effect = df.loc[eligible, "smd"].abs()
    df.loc[eligible, "contribution"] = np.sqrt(
        percentile_score(raw).clip(lower=EPS) * percentile_score(effect).clip(lower=EPS)
    )
    df["heldout_rank"] = df.contribution.rank(ascending=False, method="min")
    df.sort_values("heldout_rank").to_csv(out / f"{pop.name}_gene_omissions.csv")
    return df


def age_and_control_analysis(inputs, out, gh, mgi):
    """Run fixed analysis settings on the original population and two cartilage-dominant populations."""
    base = pd.read_csv(out / "cells.csv.gz")
    genes = pd.read_csv(out / "features.csv").iloc[:, 0].tolist()
    X = sparse.load_npz(out / "logcounts.npz")
    keep, lineage = subset_lineages(X, genes, base, "E15", out)
    Xp, gp, mp = read_postnatal(inputs)
    kp, lp = subset_lineages(Xp, gp, mp, "P28", out)
    zmeans = pd.read_csv(out / "GSE87605_zone_means.csv", index_col=0)
    zones = training_controls(zmeans, gh, mgi)
    zones.to_csv(out / "zone_control_genes.csv", index=False)
    scoresets = {"GH": gh.loc[gh.weight.ne(0), "weight"]}
    for zone, grp in zones.groupby("signature"):
        scoresets[zone] = pd.Series(1.0, index=grp.gene)
    holdout = set(zones.gene) | set(scoresets["GH"].index)
    results = []
    sex = []
    aliasrows = []
    composition = []
    for name, x, features, m in [
        ("original", X, genes, base),
        ("E15", X[keep], genes, base.loc[keep]),
        ("P28", Xp[kp], gp, mp.loc[kp]),
    ]:
        pop = prepare_population(name, x, features, m)
        mapping = selected_mappings(gh, mgi, features)
        mapping.to_csv(out / f"{name}_symbol_mapping.csv", index=False)
        w_alias = mapping.dropna(subset=["target"]).groupby("target").weight.sum()
        for label, weights in {**scoresets, "GH_alias": w_alias}.items():
            direction = {"resting": -1, "hypertrophic": 1}.get(label, 0)
            res = signature_test(
                pop,
                weights,
                label,
                701 + list(scoresets.keys()).index(label) if label in scoresets else 705,
                out,
                direction,
            )
            results.append(res)
            if label == "GH":
                gh_res = res
        # Simultaneous omission tests the graph's dependence on the input programs.
        if name != "original":
            hp = prepare_population(name + "_heldout", x, features, m, holdout | set(w_alias.index))
            for label in scoresets:
                results.append(
                    signature_test(
                        hp,
                        scoresets[label],
                        label,
                        711,
                        out,
                        {"resting": -1, "hypertrophic": 1}.get(label, 0),
                    )
                )
            del hp
        bounds = (gh_res["interior_lower"], gh_res["interior_upper"])
        if name == "original":
            historical_fit = fit_response_window(base.pseudotime, base.gh_response_score, n_boot=0)
            bounds = historical_fit["window_lower"], historical_fit["window_upper"]
        meta, sx = expression_evidence(pop, bounds, out)
        sex.append(sx)
        meta["gh_response_score"] = weighted_gh_score(x, features, gh)[0]
        if name == "original":
            meta["screen"] = lineage
            # Restore actual submitted labels to document the forced labels of excluded cells.
            assert np.array_equal(meta.cell_state.to_numpy(), base.cell_state.to_numpy())
            pd.crosstab([meta.screen, meta.cell_state], meta.interval).to_csv(
                out / "original_interval_lineages.csv"
            )
        counts = (
            meta.loc[meta.pseudotime.notna()]
            .groupby(["cell_state", "interval"])
            .size()
            .unstack(fill_value=0)
        )
        for state, row in counts.iterrows():
            composition.append(
                dict(
                    atlas=name,
                    state=state,
                    inside=row.get(True, 0),
                    outside=row.get(False, 0),
                    q25=meta.loc[meta.cell_state.eq(state), "pseudotime"].quantile(0.25),
                    median=meta.loc[meta.cell_state.eq(state), "pseudotime"].median(),
                    q75=meta.loc[meta.cell_state.eq(state), "pseudotime"].quantile(0.75),
                )
            )
        if name != "P28":
            leave_gene_out(pop, gh, out, historical=name == "original")
        if name == "E15":
            response_window_sensitivity(x, features, meta, gh).to_csv(
                out / "retained_size.csv", index=False
            )
            signature_weighting_sensitivity(x, features, meta, gh).to_csv(
                out / "retained_weights.csv", index=False
            )
            pseudotime_root_knn_robustness(
                pop.pcs, meta.filter(regex="^score_"), meta.gh_response_score, meta.pseudotime
            ).to_csv(out / "retained_graph.csv", index=False)
        ref_ids = gene_lookup(features)
        for cutoff in (0, 0.01, 0.05, 0.10):
            w = scoresets["GH"]
            sel = [
                g
                for g in w.index
                if g in ref_ids and pop.eligible[ref_ids[g]] and pop.detection[ref_ids[g]] >= cutoff
            ]
            ids = np.array([ref_ids[g] for g in sel])
            curve = standardized_curve(pop, ids, w.loc[sel].to_numpy())
            aliasrows.append(
                dict(
                    atlas=name,
                    cutoff=cutoff,
                    n=len(sel),
                    peak=pop.grid[curve.argmax()],
                    interior_amplitude=np.ptp(curve[pop.central]),
                )
            )
        marker_bins(pop, out)
        if name == "P28":
            calibrated_simulation(pop, meta.gh_response_score.to_numpy(), out)
        print("Finished", name, len(meta), flush=True)
    results = pd.DataFrame(results)
    # Adjust four age-by-zone directional checks, with graph-omission results as a separate sensitivity family.
    from statsmodels.stats.multitest import multipletests

    for family in (["E15", "P28"], ["E15_heldout", "P28_heldout"]):
        mask = results.atlas.isin(family) & results.direction_p.notna()
        results.loc[mask, "direction_holm"] = multipletests(
            results.loc[mask, "direction_p"], method="holm"
        )[1]
    results.to_csv(out / "trajectory_tests.csv", index=False)
    pd.DataFrame(sex).to_csv(out / "sex_marker_summary.csv", index=False)
    pd.DataFrame(aliasrows).to_csv(out / "detection_sensitivity.csv", index=False)
    pd.DataFrame(composition).to_csv(out / "state_composition.csv", index=False)
    return results


def marker_bins(pop, out):
    lookup = gene_lookup(pop.genes)
    names = ["Col2a1", "Pthlh", "Ihh", "Col10a1", "Mmp13"]
    tab = pd.DataFrame({g: pop.z_bins[:, lookup[g.upper()]] for g in names})
    tab["center"] = pop.centers
    counts = np.histogram(pop.meta.pseudotime.dropna(), bins=np.linspace(0, 1, 31))[0]
    tab["n"] = counts[counts > 0]
    tab.to_csv(out / f"{pop.name}_marker_bins.csv", index=False)


def calibrated_simulation(pop, score, out):
    """A conditional detectability benchmark using observed P28 support, shape and residuals."""
    t = pop.meta.pseudotime.to_numpy()
    ok = np.isfinite(t)
    t = t[ok]
    score = score[ok]
    bins = np.clip(np.digitize(t, np.linspace(0, 1, 31)) - 1, 0, 29)
    groups = [np.where(bins == i)[0] for i in np.unique(bins)]
    residuals = [score[i] - score[i].mean() for i in groups]
    ctrl = pd.read_csv(out / "P28_hypertrophic_curve.csv")
    inside = ctrl.loc[ctrl.interior]
    shape = (inside.fitted.to_numpy() - inside.fitted.min()) / np.ptp(inside.fitted)
    basis = np.interp(t, inside.pseudotime, shape)
    bin_shape = np.array([basis[i].mean() for i in groups])
    gh_curve = pd.read_csv(out / "P28_GH_curve.csv")
    scale = np.ptp(gh_curve.loc[gh_curve.interior, "fitted"])

    def simulate(amplitude, reps, rng):
        # Each column receives a vector of independently resampled bin means.
        b = np.zeros((reps, len(groups)))
        for j, r in enumerate(residuals):
            for lo in range(0, reps, 50):
                n = min(50, reps - lo)
                b[lo : lo + n, j] = (
                    rng.choice(r, (n, len(r))).mean(axis=1) + amplitude * bin_shape[j]
                )
        return np.array(
            [np.ptp(smooth_binned_profile(pop.centers, row, pop.grid[pop.central])) for row in b]
        )

    rng = np.random.default_rng(120026)
    threshold = np.quantile(simulate(0, 1000, rng), 0.95)
    rows = []
    for label, amp in [
        ("null", 0),
        ("half GH", scale / 2),
        ("GH", scale),
        ("hypertrophic control", np.ptp(inside.fitted)),
    ]:
        x = simulate(amp, 500, rng)
        hits = int((x > threshold).sum())
        ci = binomtest(hits, len(x)).proportion_ci()
        rows.append((label, amp, threshold, hits, len(x), hits / len(x), ci.low, ci.high))
    pd.DataFrame(
        rows,
        columns=[
            "signal",
            "input_interior_amplitude",
            "threshold",
            "hits",
            "trials",
            "fraction",
            "exact_lo",
            "exact_hi",
        ],
    ).to_csv(out / "calibrated_simulation.csv", index=False)


def validate_postnatal_formats(inputs, out):
    """Stream Matrix Market triplets and compare every nonzero, both axes and gene IDs with HDF5."""
    import gzip

    folder = inputs / "P28"
    paths = {
        name: folder / ("GSM8769462_" + name)
        for name in ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")
    }
    feat = pd.read_csv(
        paths["features.tsv.gz"], sep="\t", header=None, dtype=str, keep_default_na=False
    )
    bc = pd.read_csv(paths["barcodes.tsv.gz"], header=None, dtype=str).iloc[:, 0]
    with (
        (inputs / DATA_P28).open("rb") as source,
        h5py.File(source, "r") as h,
        gzip.open(paths["matrix.mtx.gz"], "rt") as f,
    ):
        m = h["matrix"]
        assert np.array_equal(bc, m["barcodes"].asstr()[:])
        for j, col in enumerate(["id", "name", "feature_type"]):
            assert np.array_equal(feat[j].to_numpy(), m["features/" + col].asstr()[:])
        line = next(f)
        while line.startswith("%"):
            line = next(f)
        nr, nc, nnz = map(int, line.split())
        assert (nr, nc) == tuple(m["shape"][:]) and nnz == len(m["data"])
        ptr = m["indptr"][:]
        offset = 0
        for tab in pd.read_csv(f, sep=" ", header=None, chunksize=500000, dtype=np.int32):
            a = tab.to_numpy()
            end = offset + len(a)
            assert np.array_equal(a[:, 0] - 1, m["indices"][offset:end])
            assert np.array_equal(
                a[:, 1] - 1, np.searchsorted(ptr[1:], np.arange(offset, end), side="right")
            )
            assert np.array_equal(a[:, 2], m["data"][offset:end])
            offset = end
        assert offset == nnz
    pd.DataFrame(
        [dict(features=nr, barcodes=nc, nonzeros=nnz, axes_agree=True, counts_agree=True)]
    ).to_csv(out / "P28_format_audit.csv", index=False)


def anatomical_transfer(inputs, out):
    """Test the two newborn signatures on physically dissected zones within each P9 donor."""
    geo = parse_geo_family_soft(inputs / "GSE14007_family.soft.gz")
    p2g = platform_probe_to_gene(geo["platforms"]["GPL1261"]["table"]).str.upper()
    samples = pd.read_csv(out / "GSE14007_samples.csv")
    features = pd.read_csv(out / "zone_control_genes.csv")
    arrays = {}
    for gsm, info in geo["samples"].items():
        a = info["table"].set_index("ID_REF")
        signal = pd.to_numeric(a[extract_value_column(a)], errors="coerce")
        arrays[gsm] = (signal / signal.median()).groupby(p2g).median()
    matrix = pd.DataFrame(arrays)
    rows = []
    for donor in ["1", "2"]:
        s = samples[
            samples.zone_text.str.contains("mouse " + donor, case=False)
            & samples.zone.isin(["resting", "proliferative", "hypertrophic"])
        ]
        x = matrix[s["sample"]].copy()
        x.columns = s.zone
        x = x.div(x.sum(axis=1), axis=0)
        for label, labels in features.groupby("signature"):
            selected = x.reindex(labels.gene).dropna()
            for zone in x.columns:
                rows.append((donor, label, zone, len(selected), selected[zone].mean()))
    pd.DataFrame(
        rows, columns=["donor", "signature", "zone", "n_genes", "mean_specificity"]
    ).to_csv(out / "P9_control_transfer.csv", index=False)


def marker_composition_test(out):
    """Condition on count-depth deciles and permute marker evidence within each library."""
    rows = []
    for atlas, category in [("E15", "Xist only"), ("P28", "multiple Y")]:
        d = (
            pd.read_csv(out / f"{atlas}_cells.csv.gz")
            .dropna(subset=["quartile"])
            .reset_index(drop=True)
        )
        q = d.quartile.to_numpy(int)
        flag = d.evidence.eq(category).to_numpy()
        sizes = np.bincount(q)
        groups = list(d.groupby("depth_bin").indices.values())

        def spread(mask):
            proportions = np.bincount(q, weights=mask, minlength=4) / sizes
            return np.ptp(proportions)

        obs = spread(flag)
        rng = np.random.default_rng(7721)
        hits = 0
        for _ in range(999):
            perm = flag.copy()
            for i in groups:
                perm[i] = rng.permutation(flag[i])
            hits += spread(perm) >= obs
        rows.append((atlas, category, len(d), obs, (hits + 1) / 1000))
    pd.DataFrame(
        rows, columns=["atlas", "evidence", "reachable", "quartile_range", "depth_conditional_p"]
    ).to_csv(out / "marker_composition_test.csv", index=False)


def analysis_figures(out):
    """Regenerate each revised panel directly from this run's tables."""
    tests = pd.read_csv(out / "trajectory_tests.csv").set_index(["atlas", "signature"])
    palette = {"GH": "#753f28", "resting": "#167b70", "hypertrophic": "#7353a0"}
    labels = {
        "original": "Submitted E15 population",
        "E15": "E15 cartilage-dominant",
        "P28": "P28 cartilage-dominant",
    }

    def read_curve(age, sig):
        return pd.read_csv(out / f"{age}_{sig}_curve.csv")

    def caption_style(ax):
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)

    f, axes = plt.subplots(
        2,
        3,
        figsize=(10, 5.3),
        layout="constrained",
        sharex="col",
        gridspec_kw={"height_ratios": [1.35, 1]},
    )
    for j, age in enumerate(labels):
        m = pd.read_csv(out / f"{age}_cells.csv.gz")
        if age == "original":
            h = pd.read_csv(out / "cells.csv.gz")
            fit = fit_response_window(h.pseudotime, h.gh_response_score, n_boot=0)
            x, y = fit["grid"], fit["smooth"]
        else:
            c = read_curve(age, "GH")
            x, y = c.pseudotime, c.fitted
        a, b = axes[:, j]
        bounds = m.pseudotime.quantile([0.1, 0.9])
        a.axvspan(*bounds, facecolor="#e5edeb")
        a.plot(x, y, color=palette["GH"], lw=1.8)
        pk = int(np.argmax(y))
        a.scatter(np.asarray(x)[pk], np.asarray(y)[pk], color=palette["GH"], s=24, zorder=3)
        a.set(title=labels[age], ylabel="Signed projection" if j == 0 else "")
        b.hist(
            m.pseudotime.dropna(),
            bins=np.linspace(0, 1, 31),
            color="#59869b",
            edgecolor="white",
            linewidth=0.3,
        )
        b.set(xlabel="Graph pseudotime", ylabel="Number of cells" if j == 0 else "", xlim=(0, 1))
        for ax in [a, b]:
            caption_style(ax)
    export_panel(f, "lineage_profiles")
    f, axes = plt.subplots(2, 3, figsize=(10.5, 5.2), layout="constrained", sharex=True)
    for i, age in enumerate(["E15", "P28"]):
        for j, sig in enumerate(["resting", "hypertrophic", "GH"]):
            ax = axes[i, j]
            c = read_curve(age, sig)
            z = tests.loc[(age, sig)]
            mid = c[c.interior]
            ax.plot(c.pseudotime, c.fitted, color="#a9acad", lw=1)
            ax.plot(mid.pseudotime, mid.fitted, color=palette[sig], lw=2.4)
            ax.set_title(
                f"{age}   {sig.capitalize() if sig != 'GH' else 'Whole-bone GH'}", fontsize=11
            )
            ax.set(
                xlabel="Graph pseudotime" if i else "", ylabel="Signed projection" if j == 0 else ""
            )
            ptxt = "<0.001" if z.matched_p < 0.001 else f"= {z.matched_p:.3f}"
            ax.text(
                0.03,
                0.04,
                f"Interior amplitude {z.amplitude:.2f}\nMatched p {ptxt}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=2),
            )
            caption_style(ax)
    export_panel(f, "zone_controls")
    f, axes = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained")
    for ax, age in zip(axes, labels):
        b = pd.read_csv(out / f"{age}_marker_bins.csv")
        for g in ["Col2a1", "Pthlh", "Ihh", "Col10a1", "Mmp13"]:
            ax.plot(b.center, b[g], label=g, lw=1.25)
        ax.set(title=labels[age], xlabel="Graph pseudotime")
        caption_style(ax)
    axes[0].set_ylabel("Mean standardized expression")
    axes[-1].legend(fontsize=7, frameon=False)
    export_panel(f, "maturation_markers")
    f, ax = plt.subplots(figsize=(5.9, 4.2), layout="constrained")
    q = pd.read_csv(out / "expanded_target_qc.csv").dropna(subset=["estimated_log2fc"])
    for time, symbol, color in [("6h", "o", "#277584"), ("24h", "s", "#ac5938")]:
        u = q[q.time == time]
        ax.scatter(
            u.published_log2fc, u.estimated_log2fc, label=time, marker=symbol, color=color, s=35
        )
    r = 1.08 * max(abs(q.published_log2fc).max(), abs(q.estimated_log2fc).max())
    ax.plot([-r, r], [-r, r], ls="--", color="0.55")
    ax.axhline(0, color=".75", lw=0.6)
    ax.axvline(0, color=".75", lw=0.6)
    for gene, time in [("IGF1", "6h"), ("STAT6", "24h"), ("BMP7", "24h")]:
        v = q[(q.gene == gene) & (q.time == time)].iloc[0]
        ax.annotate(
            gene,
            (v.published_log2fc, v.estimated_log2fc),
            xytext=(4, -9),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        xlabel="Published log₂ fold change",
        ylabel="Reconstructed log₂ fold change",
        xlim=(-r, r),
        ylim=(-r, r),
    )
    ax.legend(frameon=False)
    caption_style(ax)
    export_panel(f, "array_audit")
    f, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), layout="constrained")
    d = pd.read_csv(out / "original_gene_omissions.csv").dropna(
        subset=["original_rank", "heldout_rank"]
    )
    axes[0].scatter(d.original_rank, d.heldout_rank, s=11, color="#407082")
    axes[0].plot([1, len(d)], [1, len(d)], color="#888", lw=0.8)
    axes[0].set(
        xlabel="Submitted contribution rank", ylabel="Held-out gene rank", title="Original interval"
    )
    d = (
        pd.read_csv(out / "E15_gene_omissions.csv")
        .dropna(subset=["heldout_rank"])
        .nsmallest(10, "heldout_rank")
        .sort_values("contribution")
    )
    axes[1].barh(d.gene, d.contribution, color="#497b75")
    axes[1].set(title="E15 cartilage-dominant interval", xlabel="Contribution score")
    for ax in axes:
        caption_style(ax)
    export_panel(f, "heldout_ranking")
    f, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1.4]}
    )
    colors = ["#407e94", "#ce7647", "#8f6da7", "#c9ced0"]
    for ax, age in zip(axes[:2], ["E15", "P28"]):
        df = pd.read_csv(out / f"{age}_sex_by_quartile.csv", index_col=0)
        cols = ["Xist only", "multiple Y", "mixed", "indeterminate"]
        d = df[cols].div(df.sum(axis=1), axis=0)
        bottom = np.zeros(len(d))
        for col, color in zip(cols, colors):
            ax.bar(np.arange(1, 5), d[col], bottom=bottom, label=col, color=color)
            bottom += d[col].to_numpy()
        ax.set(title=age, xlabel="Pseudotime quartile", xticks=[1, 2, 3, 4], ylim=(0, 1))
    axes[0].set_ylabel("Fraction of cells")
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7, frameon=False
    )
    s = pd.read_csv(out / "zonal_sex_probes.csv")
    s = s[s.accession == "GSE14007"]
    mat = s.pivot(index="gene", columns="sample", values="relative_to_array_median").reindex(
        SEX_PANEL
    )
    im = axes[2].imshow(
        np.log2(mat.clip(lower=0.01)), vmin=-7, vmax=7, cmap="RdBu_r", aspect="auto"
    )
    axes[2].set(
        yticks=range(5),
        yticklabels=SEX_PANEL,
        xticks=range(8),
        xticklabels=["R1", "R2", "P1", "P2", "M1", "M2", "H1", "H2"],
        title="P9 zonal arrays",
        xlabel="Dissected zone and mouse",
    )
    f.colorbar(im, ax=axes[2], label="log₂(signal / array median)", shrink=0.8)
    export_panel(f, "sex_evidence")


FIGURE_STEMS = {
    "1": "time",
    "2": "lineage_profiles",
    "3": "zone_controls",
    "S1": "maturation_markers",
    "S2": "embedding",
    "S3": "array_audit",
    "S4": "fig14_h1_developmental_structure_null",
    "S5": "fig15_h2_zonal_localization_null",
    "S6": "fig08_response_window_sensitivity",
    "S7": "fig11_signature_weighting_sensitivity",
    "S8": "fig12_pseudotime_root_knn_robustness",
    "S9": "heldout_ranking",
    "S10": "pairing",
    "S11": "sex_evidence",
    "S12": "atlas",
}


def publication_tables(out, h1):
    """Build both printed tables and the exact S4 numbers from the run underlying the panels."""
    from scipy.stats import binomtest

    f = pd.read_csv(out / "state_composition.csv")
    old = pd.read_csv(out / "original_interval_lineages.csv")
    labels = ["resting", "proliferative", "prehypertrophic", "hypertrophic"]
    orig = f[f.atlas == "original"].set_index("state").reindex(labels)
    e = f[f.atlas == "E15"].set_index("state").reindex(labels)
    excluded = old[old.screen != "cartilage"].groupby("cell_state")["True"].sum()
    counts = pd.DataFrame(
        {
            "submitted_inside": orig.inside,
            "removed_from_submitted_inside": excluded,
            "retained_from_submitted_inside": orig.inside - excluded,
            "E15_new_inside": e.inside,
            "E15_new_outside": e.outside,
            "E15_median": e["median"],
            "E15_q25": e.q25,
            "E15_q75": e.q75,
        }
    )
    counts.index.name = "state"
    counts.to_csv(out / "table1.csv")
    alltests = pd.read_csv(out / "trajectory_tests.csv")
    primary = alltests[
        alltests.atlas.isin(["E15", "P28"])
        & alltests.signature.isin(["GH", "hypertrophic", "resting"])
    ]
    primary.to_csv(out / "table2.csv", index=False)
    # Recount exceedances independently of the plotting call, using its archived draws.
    kmap = {"Weight": "weight_permutation", "Matched": "matched_random_genes"}
    summ = h1["summary"]
    drawn = h1["null_stats"]
    report = []
    for label, key in kmap.items():
        group = drawn.loc[drawn.null_type.eq(key), "amplitude"]
        exceed = int((group >= summ["observed_amplitude"]).sum())
        value = (1 + exceed) / (1 + len(group))
        field = (
            "weight_permutation_p_amplitude" if label == "Weight" else "matched_gene_p_amplitude"
        )
        assert value == summ[field]
        report.append(
            dict(null=key, draws=len(group), exceedances=exceed, p=value, display=f"{value:.4f}")
        )
    pd.DataFrame(report).to_csv(out / "H1_figure_values.csv", index=False)
    (out / "reported_values.tex").write_text(
        "".join(
            "\\newcommand{\\HOne" + key + "P}{" + row["display"] + "}\n"
            for key, row in zip(kmap, report)
        ),
        encoding="utf-8",
    )
    noise = binomtest(17, 200, 0.05)
    ci = noise.proportion_ci()
    pd.DataFrame(
        [
            dict(
                hits=17,
                trials=200,
                fraction=0.085,
                exact_lo=ci.low,
                exact_hi=ci.high,
                nominal=0.05,
                upper_binomial_probability=binomtest(17, 200, 0.05, alternative="greater").pvalue,
            )
        ]
    ).to_csv(out / "original_simulation_reference.csv", index=False)


def verify_revision(out, figdir, root):
    """Check observed results against rounded publication values and all export signatures."""
    cums = []

    def audit(name, obs, ref, tol=0):
        obs = float(obs)
        cums.append(
            dict(
                quantity=name,
                computed=obs,
                reported=ref,
                tolerance=tol,
                passed=abs(obs - ref) <= tol,
            )
        )

    table = pd.read_csv(out / "table1.csv")
    for k, v in {
        "submitted_inside": 1833,
        "removed_from_submitted_inside": 782,
        "E15_new_inside": 425,
        "E15_new_outside": 5032,
    }.items():
        audit(k, table[k].sum(), v)
    p = pd.read_csv(out / "trajectory_tests.csv").set_index(["atlas", "signature"])
    for age, sig, key, value in [
        ("E15", "GH", "matched_p", 0.5445),
        ("P28", "GH", "matched_p", 0.7542),
        ("E15", "GH_alias", "matched_p", 0.6334),
        ("P28", "GH_alias", "matched_p", 0.9021),
        ("P28", "hypertrophic", "direction_holm", 0.0040),
        ("E15", "hypertrophic", "direction_holm", 0.7493),
        ("P28", "resting", "direction_holm", 1.0),
        ("E15", "resting", "direction_holm", 1.0),
        ("P28_heldout", "hypertrophic", "direction_holm", 0.0040),
        ("E15", "GH", "amplitude", 0.2609),
        ("P28", "GH", "amplitude", 0.1867),
    ]:
        audit(f"{age} {sig} {key}", p.loc[(age, sig), key], value, 0.0001)
    for age, n in [("original", 123), ("E15", 120)]:
        t = pd.read_csv(out / f"{age}_gene_omissions.csv").dropna(subset=["heldout_rank"])
        audit(age + " ranked held-out genes", len(t), n)
        for gene, r in [("WNT5A", 1), ("MEG3", 2)]:
            audit(
                age + " " + gene + " omission rank",
                t.set_index("gene").loc[gene, "heldout_rank"],
                r,
            )
    q = pd.read_csv(out / "expanded_target_qc.csv").dropna(subset=["estimated_log2fc"])
    audit("Evaluable published target/time pairs", len(q), 19)
    audit(
        "Direction agreement",
        np.sum(np.sign(q.estimated_log2fc) == np.sign(q.published_log2fc)),
        18,
    )
    audit(
        "Absolute effect attenuation",
        np.sum(q.estimated_log2fc.abs() < q.published_log2fc.abs()),
        19,
    )
    m = pd.read_csv(out / "E15_symbol_mapping.csv")
    for key, n in [("direct", 153), ("unique_alias", 107), ("ambiguous", 2), ("unmapped", 38)]:
        audit("E15 symbol " + key, m.rule.eq(key).sum(), n)
    dat = pd.read_csv(out / "P28_format_audit.csv").iloc[0]
    audit("HDF5 and Matrix Market identical nonzero entries", dat.nonzeros, 45599673)
    sx = pd.read_csv(out / "sex_marker_summary.csv").set_index("atlas")
    audit("P28 multi-Y without Xist", sx.loc["P28", "multiple Y"], 5787)
    audit("E15 Xist only", sx.loc["E15", "Xist only"], 5339)
    for label, n in [("null", 15), ("half GH", 134), ("GH", 361), ("hypertrophic control", 500)]:
        sim = pd.read_csv(out / "calibrated_simulation.csv", keep_default_na=False).set_index(
            "signal"
        )
        audit("Signal simulation " + label, sim.loc[label, "hits"], n)
    old = pd.read_csv(out / "H1_figure_values.csv")
    for i, v in enumerate([0.0719, 0.0869]):
        audit("Figure S4 " + old.iloc[i]["null"], old.iloc[i].p, v, 0.00005)
    values = pd.DataFrame(cums)
    values.to_csv(out / "revision_audit.csv", index=False)
    assert values.passed.all(), values.loc[~values.passed].to_string(index=False)
    idx = []
    for name, stem in FIGURE_STEMS.items():
        for ext, header in [("pdf", b"%PDF"), ("png", b"\x89PNG"), ("eps", b"%!PS")]:
            path = figdir / f"{stem}.{ext}"
            with path.open("rb") as handle:
                assert handle.read(len(header)) == header, path
            idx.append(dict(figure=name, filename=path.name, bytes=path.stat().st_size))
    pd.DataFrame(idx).to_csv(root / "figure_index.csv", index=False)
    print(f"Passed {len(values)} revision checks and verified {len(idx)} figure exports.")
