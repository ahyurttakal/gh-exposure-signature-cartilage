"""Run the complete manuscript analysis from the command line."""

from ghir_methods import (
    run_regression_checks,
    CANONICAL_GH_IGF,
    EPS,
    KMeans,
    Path,
    UnivariateSpline,
    _rank_quantile_bins,
    _sparse_gene_moments,
    _sparse_standardized_weighted_score,
    analyze_gse3689,
    analyze_single_cell,
    annotate_states,
    candidate_ranking_sensitivity,
    cross_species_conservation,
    export_panel,
    fig_candidate_rank_robustness,
    fig_gse3689_preprocessing_qc,
    fig_pseudotime_root_knn_robustness,
    fig_response_window_null_test,
    fig_response_window_sensitivity,
    fig_signature_weighting_sensitivity,
    fig_zonal_localization_null_test,
    fit_response_window,
    smooth_binned_profile,
    gene_lookup,
    graph_geodesic_pseudotime,
    gse3689_published_target_qc,
    gzip,
    hashlib,
    io,
    json,
    np,
    os,
    parse_geo_family_soft,
    pd,
    plt,
    process_zonal_geo,
    pseudotime_root_knn_robustness,
    response_window_null_test,
    response_window_sensitivity,
    select_hvg_and_pca,
    shutil,
    signature_weighting_sensitivity,
    sparse,
    spearmanr,
    time,
    urllib,
    weighted_gh_score,
    zonal_gh_propensity,
    zonal_localization_null_test,
    zone_specificity,
)

from revision_analysis import (
    age_and_control_analysis,
    array_marker_audit,
    anatomical_transfer,
    marker_composition_test,
    validate_postnatal_formats,
    analysis_figures,
    publication_tables,
    verify_revision,
)


def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


def run_analysis(inputs, root, versions):
    """Recompute all results from the cache already checked by the entry point."""
    import sys
    from run_settings import display_path

    out, figdir = root / "tables", root / "figures"
    for folder in (out, figdir):
        folder.mkdir(parents=True, exist_ok=True)
    os.environ["GHIR_FIGURES"] = str(figdir)
    run_regression_checks()

    annotation_dir = Path(__file__).resolve().parent / "annotations"
    ortholog_path = annotation_dir / "ortholog_pairs.csv.gz"
    ortholog_csv = gzip.decompress(ortholog_path.read_bytes())
    if (
        hashlib.sha256(ortholog_csv).hexdigest()
        != "dd584b960ebde57bf64c5aa55caa96c95e09835385b03db00ebac2fc35b79e21"
    ):
        raise ValueError("Frozen ortholog identifiers failed the checksum.")
    ortholog_pairs = pd.read_csv(io.BytesIO(ortholog_csv))
    assert ortholog_pairs.mouse_gene.is_unique and ortholog_pairs.human_gene.is_unique
    print(len(ortholog_pairs), "archived one-to-one identifiers")

    data = inputs
    gh, md = analyze_gse3689(parse_geo_family_soft(data / "GSE3689_family.soft.gz"))
    gh.to_csv(out / "genes.csv")
    md.to_csv(out / "arrays.csv", index=False)
    print("FDR", gh.shape, (gh.filter(regex="^fdr_") < 0.05).sum(), flush=True)
    print(
        "concordance",
        gh[["log2fc_6h", "log2fc_24h"]].dropna().shape,
        gh.log2fc_6h.corr(gh.log2fc_24h, method="spearman"),
        flush=True,
    )
    print(gse3689_published_target_qc(gh), flush=True)
    print(
        gh.reindex(sorted(CANONICAL_GH_IGF))[
            ["weight", "rank_abs_weight", "log2fc_6h", "log2fc_24h"]
        ].to_string(),
        flush=True,
    )

    print(
        "Selected genes passing each FDR cutoff:",
        (gh.loc[gh.weight != 0].filter(regex="^fdr_") < 0.05).sum(),
    )

    zones3 = ["resting", "proliferative", "hypertrophic"]
    zmeans = {}
    specs = {}
    for accession in ["GSE87605", "GSE14007", "GSE9160"]:
        z, meta = process_zonal_geo(
            parse_geo_family_soft(inputs / f"{accession}_family.soft.gz"),
            human=accession == "GSE9160",
        )
        zmeans[accession] = z
        specs[accession] = zone_specificity(z, zones3)
        z.to_csv(out / f"{accession}_zone_means.csv")
        meta.to_csv(out / f"{accession}_samples.csv", index=False)
        print(accession, z.shape, list(z.columns))

    h2 = {}
    zonal_tests = {}
    for accession, seed in [("GSE87605", 42), ("GSE14007", 43)]:
        spec = specs[accession]
        prop = zonal_gh_propensity(spec, gh)
        prop.to_csv(out / f"{accession}_burden.csv")
        result = zonal_localization_null_test(
            spec, zmeans[accession], gh, n_permutations=1000, random_state=seed
        )
        zonal_tests[accession] = result
        h2[accession] = result["summary"]
        result["null_stats"].to_csv(out / f"{accession}_null.csv", index=False)
        display(prop)
    display(pd.DataFrame(h2).T)

    atlas = analyze_single_cell(data / "sc", gh)
    atlas["meta"].to_csv(out / "cells.csv.gz", index=False)
    pd.Series(atlas["genes"]).to_csv(out / "features.csv", index=False)
    sparse.save_npz(out / "logcounts.npz", atlas["X"])
    print("cells", len(atlas["meta"]), len(atlas["gh_genes_used"]), flush=True)
    print(
        {
            k: a
            for k, a in fit_response_window(
                atlas["meta"].pseudotime, atlas["meta"].gh_response_score, n_boot=0
            ).items()
            if np.isscalar(a)
        },
        flush=True,
    )

    h1 = response_window_null_test(
        atlas["X"], atlas["genes"], atlas["meta"], gh, n_permutations=1000, random_state=42
    )
    h1["null_stats"].to_csv(out / "H1_original_null.csv", index=False)
    display(pd.Series(h1["summary"]))

    d = out
    fig = figdir
    X = sparse.load_npz(d / "logcounts.npz")
    m = pd.read_csv(d / "cells.csv.gz")
    genes = pd.read_csv(d / "features.csv").iloc[:, 0].tolist()
    gh = pd.read_csv(d / "genes.csv", index_col=0)
    lookup = gene_lookup(genes)
    t = m.pseudotime.to_numpy()
    score = m.gh_response_score.to_numpy()
    state = m.cell_state.to_numpy()
    zones = ["resting", "proliferative", "prehypertrophic", "hypertrophic"]
    mu, sd, det = _sparse_gene_moments(X)
    selected = [g for g in gh.index[gh.weight != 0] if g in lookup]
    active = [g for g in selected if sd[lookup[g]] > EPS]
    ids = np.array([lookup[g] for g in active])
    w = gh.loc[active, "weight"].to_numpy()
    print(
        "coverage selected, mapped, nonconstant, prevalence<5%",
        sum(gh.weight != 0),
        len(selected),
        len(ids),
        (det[ids] < 0.05).sum(),
    )
    win = fit_response_window(t, score, n_boot=200)
    lo, hi = win["window_lower"], win["window_upper"]
    inside = (t >= lo) & (t <= hi)
    print("finite/interval", np.isfinite(t).sum(), inside.sum(), "window/peak", lo, hi, win["peak"])
    print("state counts", pd.crosstab(m.cell_state, inside))
    print("outside finite", pd.Series(state[np.isfinite(t) & ~inside]).value_counts())
    print("state pseudotime", m.groupby("cell_state").pseudotime.quantile([0.25, 0.5, 0.75]))
    print(
        "module means",
        m.loc[np.isfinite(t)]
        .groupby(inside[np.isfinite(t)])[["score_gh_transduction", "score_igf_axis"]]
        .mean(),
    )

    composition = pd.crosstab(m.cell_state, pd.Series(inside, name="inside"))
    composition.to_csv(out / "table1_counts.csv")
    m.groupby("cell_state").pseudotime.quantile([0.25, 0.5, 0.75]).to_csv(
        out / "table1_pseudotime.csv"
    )
    state_table = pd.DataFrame({"inside_n": composition[True], "outside_n": composition[False]})
    for side in ("inside", "outside"):
        counts_col = state_table[side + "_n"]
        state_table[side + "_pct"] = 100 * counts_col / counts_col.sum()
    quartiles = m.groupby("cell_state").pseudotime.quantile([0.25, 0.5, 0.75]).unstack()
    quartiles.columns = ["q25", "median", "q75"]
    state_table.join(quartiles).to_csv(out / "table1.csv")

    def score_subset(keep):
        return _sparse_standardized_weighted_score(X, ids[keep], w[keep], mu, sd)

    for cutoff in [0.01, 0.05, 0.1]:
        keep = det[ids] >= cutoff
        y = score_subset(keep)
        s = fit_response_window(t, y, n_boot=0)
        print("detection", cutoff, keep.sum(), spearmanr(score, y).statistic, s["peak"])
    for g in ["WNT5A", "KITL", "PTMA", "SURF6", "MEG3"]:
        if g not in active:
            continue
        y = score_subset(np.array(active) != g)
        s = fit_response_window(t, y, n_boot=0)
        split = (t >= s["window_lower"]) & (t <= s["window_upper"])
        print(
            "leave-one-out",
            g,
            s["peak"],
            s["window_lower"],
            s["window_upper"],
            (split & inside).sum() / (split | inside).sum(),
        )

    # Equal-width bin sufficient statistics. Weighted gene sums commute with bin averaging.
    valid = np.isfinite(t)
    i = np.flatnonzero(valid)
    bins = np.clip(np.floor(t[valid] * 30).astype(int), 0, 29)
    nb = np.bincount(bins, minlength=30)
    seen = nb > 0
    B = sparse.csr_matrix((1 / nb[bins], (bins, i)), shape=(30, len(t)))
    Z = ((B @ X).toarray()[seen] - mu) / np.maximum(sd, EPS)
    centers = (np.flatnonzero(seen) + 0.5) / 30
    grid = np.linspace(centers.min(), centers.max(), 300)
    interior = np.quantile(t[valid], [0.1, 0.9])
    middle = (grid >= interior[0]) & (grid <= interior[1])

    def curve(ids_, weights):
        y = Z[:, ids_] @ weights / np.linalg.norm(weights)
        return smooth_binned_profile(centers, y, grid)

    def metrics(y):
        return np.array([np.ptp(y), np.ptp(y[middle])])

    observed = metrics(curve(ids, w))
    rng = np.random.default_rng(42017)
    eligible = (sd > EPS) & (det > 0)
    variability = np.sqrt(
        np.average((Z - np.average(Z, axis=0, weights=nb[seen])) ** 2, axis=0, weights=nb[seen])
    )
    strata = np.zeros(len(genes), int)
    for j, a in enumerate([mu, det, variability]):
        strata[eligible] += 4**j * _rank_quantile_bins(a[eligible], 4)
    background = eligible.copy()
    background[ids] = False
    pools = [
        (np.flatnonzero(background & (strata == k)), int((strata[ids] == k).sum()))
        for k in np.unique(strata[ids])
    ]
    assert all(len(pool) >= n for pool, n in pools)
    nulls = [[], []]
    for _ in range(1000):
        nulls[0].append(metrics(curve(ids, rng.permutation(w))))
        sample = np.concatenate([rng.choice(pool, n, replace=False) for pool, n in pools])
        nulls[1].append(metrics(curve(sample, rng.permutation(w))))
    print(
        "interior",
        interior,
        "observed",
        observed,
        "new p (full, interior)",
        [(1 + (np.array(a) >= observed).sum(axis=0)) / 1001 for a in nulls],
    )
    print("endpoint bin counts", nb.tolist())

    revision_null_p = np.stack([(1 + (np.array(a) >= observed).sum(axis=0)) / 1001 for a in nulls])
    pd.DataFrame(
        np.hstack(nulls),
        columns=["shuffle_full", "shuffle_interior", "matched_full", "matched_interior"],
    ).to_csv(out / "H1_revision_null.csv", index=False)

    # Cluster-based lineage sensitivity. Exclude clusters where a competing positive
    # marker panel exceeds the cartilage panel in mean log-normalized expression.
    pcs, _, _ = select_hvg_and_pca(X, genes)
    cl = KMeans(n_clusters=18, n_init=10, random_state=42).fit_predict(pcs)
    panels = {
        "cartilage": ["Col2a1", "Acan", "Sox9", "Comp", "Col9a1"],
        "immune": ["Ptprc", "Lyz2", "Tyrobp"],
        "endothelial": ["Pecam1", "Cdh5", "Kdr"],
        "bone/perichondrium": ["Col1a1", "Col1a2", "Sp7", "Bglap"],
        "interzone": ["Gdf5", "Wnt9a", "Prg4"],
        "epithelial": ["Epcam", "Krt8", "Krt18"],
    }
    pan = pd.DataFrame(
        {
            k: np.asarray(
                X[:, [lookup[g.upper()] for g in gs if g.upper() in lookup]].mean(axis=1)
            ).ravel()
            for k, gs in panels.items()
        }
    )
    by = pan.groupby(cl).mean()
    by["assignment"] = by.idxmax(axis=1)
    by["n"] = np.bincount(cl)
    by.to_csv(d / "lineage_clusters.csv")
    keep = np.isin(cl, by.index[by.assignment == "cartilage"])
    Xk = X[keep]
    pck, _, _ = select_hvg_and_pca(Xk, genes)
    states, _ = annotate_states(Xk, genes)
    tk, _, _ = graph_geodesic_pseudotime(pck, states)
    yk, _ = weighted_gh_score(Xk, genes, gh)
    sk = fit_response_window(tk, yk, n_boot=0)
    print(
        "lineage sensitivity",
        by.to_string(),
        keep.sum(),
        np.isfinite(tk).sum(),
        sk["peak"],
        sk["window_lower"],
        sk["window_upper"],
    )
    retained = pd.DataFrame({"barcode": m.barcode, "cluster": cl, "cartilage_cluster": keep})
    retained.to_csv(d / "lineage_membership.csv.gz", index=False)

    fig1, ax = plt.subplots(2, 2, figsize=(9, 6.4), layout="constrained")
    a, b, c, e = ax.flat
    a.plot(win["grid"], win["smooth"], color="#174c69")
    a.fill_between(win["grid"], win["ci_low"], win["ci_high"], color="#a5c6d7")
    a.plot(win["grid"], win["smooth"], color="#174c69")
    a.axvspan(lo, hi, color="#cc8e36", alpha=0.2)
    a.set(title="A   Projected score", ylabel="Signed similarity score")
    b.bar(centers, nb[seen], width=0.03, color="#477787")
    b.set(title="B   Cell support", ylabel="Cells per bin")
    colors = ["#397093", "#e5a140", "#3b8d70", "#865c8c"]
    for lab, col in zip(zones, colors):
        c.hist(
            t[(state == lab) & valid],
            bins=np.linspace(0, 1, 21),
            histtype="step",
            linewidth=1.6,
            label=lab,
            color=col,
        )
    c.set(title="C   Marker-assigned states", ylabel="Cells per bin")
    c.legend(fontsize=7, frameon=False)
    for name, lab, col in [
        ("score_gh_transduction", "GH transduction", "#275c85"),
        ("score_igf_axis", "IGF axis", "#b25436"),
    ]:
        e.plot(centers, (B @ m[name].values)[seen], label=lab, color=col)
    e.set(title="D   Canonical module expression", ylabel="Mean standardized module score")
    e.legend(frameon=False, fontsize=8)
    for a in ax.flat:
        a.set_xlabel("Graph pseudotime")
    export_panel(fig1, "atlas")

    gk = sk["grid"]
    fk = sk["smooth"]
    bounds = np.quantile(tk[np.isfinite(tk)], [0.1, 0.9])
    mk = (gk >= bounds[0]) & (gk <= bounds[1])
    im = np.flatnonzero(mk)[np.argmax(fk[mk])]
    midcurve = curve(ids, w)[middle]
    mi = midcurve.argmax()
    mg = grid[middle]
    above = midcurve >= midcurve.min() + 0.8 * np.ptp(midcurve)
    l = r = mi
    while l and above[l - 1]:
        l -= 1
    while r + 1 < len(mg) and above[r + 1]:
        r += 1
    print(
        "interior ref threshold (original spline)",
        mg[mi],
        mg[l],
        mg[r],
        "filtered interior peak, support",
        gk[im],
        bounds,
        np.histogram(tk, np.linspace(0, 1, 31))[0][:4],
    )

    # Use the same PC axes in both panels for a labeled atlas map and a score map.
    f, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for lab, col in zip(zones, colors):
        mask = state == lab
        a.scatter(pcs[mask, 0], pcs[mask, 1], s=2, color=col, label=lab, rasterized=True)
    a.legend(fontsize=7, markerscale=3, frameon=False)
    a.set_title("A   Highest marker module")
    lim = np.quantile(abs(score), 0.99)
    art = b.scatter(
        pcs[:, 0], pcs[:, 1], c=score, cmap="RdBu_r", vmin=-lim, vmax=lim, s=3, rasterized=True
    )
    f.colorbar(art, ax=b, label="Signed score", extend="both")
    b.set_title("B   Whole-bone signature projection")
    for ax in [a, b]:
        ax.set(xlabel="SVD component 1", ylabel="SVD component 2")
    export_panel(f, "embedding")

    cons, cs = cross_species_conservation(specs["GSE87605"], specs["GSE9160"])
    cons.to_csv(out / "cross_species.csv")
    mouse_spec = specs["GSE87605"]
    human_spec = specs["GSE9160"]
    om = mouse_spec.loc[ortholog_pairs.mouse_gene, zones3].to_numpy()
    oh = human_spec.loc[ortholog_pairs.human_gene, zones3].to_numpy()
    orth_sim = 1 - np.abs(om / om.sum(1)[:, None] - oh / oh.sum(1)[:, None]).sum(1) / 2
    ortholog_pairs.assign(similarity=orth_sim).to_csv(out / "orthologs.csv", index=False)
    print(cs, "ortholog mean", orth_sim.mean())

    data = cons.rename_axis("gene").reset_index()
    zones = ["resting", "proliferative", "hypertrophic"]
    mouse, human = [
        data[[f"{species}_{zone}" for zone in zones]].to_numpy() for species in ("mouse", "human")
    ]
    assert data.gene.is_unique and np.isfinite([mouse, human]).all()
    similarity = lambda a, b: 1 - np.abs(a - b).sum(axis=1) / 2
    assert np.allclose(similarity(mouse, human), data.conservation)
    observed = similarity(mouse, human).mean()
    rng = np.random.default_rng(271634)
    null = np.array(
        [similarity(mouse, human[rng.permutation(len(human))]).mean() for _ in range(10000)]
    )
    m, h = mouse.argmax(1), human.argmax(1)
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (m, h), 1)
    fractions = counts / counts.sum(axis=1, keepdims=True)
    margins = counts.sum(axis=0) / counts.sum()
    print(
        "n, observed, null mean, quantiles, difference, p",
        len(data),
        observed,
        null.mean(),
        np.quantile(null, [0.025, 0.975]),
        observed - null.mean(),
        (1 + (null >= observed).sum()) / (1 + len(null)),
    )
    print(
        "row fractions, human margins, overall agreement, expected agreement",
        fractions,
        margins,
        counts.trace() / counts.sum(),
        (counts.sum(axis=1) * margins).sum() / counts.sum(),
    )
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(8.7, 3.3), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]}
    )
    a.hist(null, bins=35, color="#6d96ae")
    a.axvline(observed, color="#ae4c29", lw=2, label="Observed")
    a.set(xlabel="Mean zonal similarity", ylabel="Permutations", title="A   Gene-pair shuffle")
    a.legend(frameon=False)
    delta = fractions - margins
    im = b.imshow(delta, cmap="RdBu", vmin=-0.08, vmax=0.08)
    for i, j in np.ndindex(3, 3):
        b.text(j, i, f"{100 * delta[i, j]:+.1f}", ha="center", va="center", color="black")
    b.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=["R", "P", "H"],
        yticklabels=["R", "P", "H"],
        xlabel="Human dominant zone",
        ylabel="Mouse dominant zone",
        title="B   Excess over human margin (pp)",
    )
    export_panel(fig, "pairing")
    # Separate exploratory subset: permute partners only within mapped selected genes.
    weights = gh
    used = data.gene.isin(weights.index[weights.weight != 0]).to_numpy()
    a, b = mouse[used], human[used]
    obs = similarity(a, b).mean()
    rng = np.random.default_rng(271635)
    draws = np.array([similarity(a, b[rng.permutation(len(b))]).mean() for _ in range(10000)])
    print(
        "selected subset (n, observed, null mean, 95% interval, p)",
        len(a),
        obs,
        draws.mean(),
        np.quantile(draws, [0.025, 0.975]),
        (1 + (draws >= obs).sum()) / 10001,
    )

    pd.DataFrame({"all_pairs": null, "signature_pairs": draws}).to_csv(
        out / "pairing_nulls.csv", index=False
    )

    meta = atlas["meta"]
    size_sens = response_window_sensitivity(X, genes, meta, gh)
    weight_sens = signature_weighting_sensitivity(X, genes, meta, gh)
    root_sens = pseudotime_root_knn_robustness(
        atlas["pcs"], meta.filter(regex="^score_"), meta.gh_response_score, meta.pseudotime
    )
    rank_long, rank_summary, rank_matrix = candidate_ranking_sensitivity(X, genes, meta, gh, cons)
    for name, table in [
        ("signature_size", size_sens),
        ("weights", weight_sens),
        ("root_knn", root_sens),
        ("ranks", rank_long),
        ("rank_summary", rank_summary),
        ("rank_matrix", rank_matrix),
    ]:
        table.to_csv(out / f"{name}.csv")
    display(rank_summary)
    display(rank_matrix.head())
    display(root_sens)

    t = pd.read_csv(d / "cells.csv.gz").pseudotime.dropna().to_numpy()
    bin_id = np.minimum((30 * t).astype(int), 29)
    n = np.bincount(bin_id, minlength=30)
    x = (np.flatnonzero(n) + 0.5) / 30
    g = np.linspace(*np.quantile(t, [0.1, 0.9]), 200)

    def summary(y):
        means = np.bincount(bin_id, weights=y, minlength=30)[n > 0] / n[n > 0]
        pred = UnivariateSpline(x, means, s=0.02 * len(x))(g)
        return np.ptp(pred), g[np.argmax(pred)]

    rng = np.random.default_rng(71)
    baseline = np.array([summary(rng.normal(size=len(t)))[0] for _ in range(1000)])
    threshold = np.quantile(baseline, 0.95)
    hits = []
    for amplitude in [0, 1]:
        bump = amplitude * np.exp(-0.5 * ((t - 0.60) / 0.08) ** 2)
        results = np.array([summary(bump + rng.normal(size=len(t))) for _ in range(200)])
        hits.append(int((results[:, 0] > threshold).sum()))
        print(amplitude, (results[:, 0] > threshold).sum(), np.median(abs(results[:, 1] - 0.60)))
    gh = pd.read_csv(d / "genes.csv", index_col=0)
    x = gh.log2fc_6h
    y = gh.log2fc_24h
    f, a = plt.subplots(figsize=(6.3, 3.8), layout="constrained")
    a.scatter(x, y, color="#27647c", s=5, alpha=0.25, rasterized=True)
    a.axhline(0, color=".65", lw=0.6)
    a.axvline(0, color=".65", lw=0.6)
    a.set(
        xlabel="Whole-bone GH / PBS, 6 h (log2 ratio)",
        ylabel="Whole-bone GH / PBS, 24 h (log2 ratio)",
    )
    a.text(
        0.03,
        0.96,
        f"{len(gh)} genes  |  Spearman ρ = {x.corr(y, method='spearman'):.3f}",
        va="top",
        transform=a.transAxes,
    )
    export_panel(f, "time")

    target_qc = gse3689_published_target_qc(gh)
    target_qc.to_csv(out / "published_target_qc.csv", index=False)
    fig_response_window_null_test(h1)
    fig_zonal_localization_null_test(zonal_tests["GSE87605"])

    checks = []

    def check(name, computed, reported, tolerance):
        computed = float(computed)
        checks.append(
            dict(
                quantity=name,
                computed=computed,
                reported=reported,
                tolerance=tolerance,
                passed=bool(abs(computed - reported) <= tolerance),
            )
        )

    for key, num in [("6h", 0), ("24h", 2), ("combined", 909)]:
        check("BH discoveries " + key, (gh["fdr_" + key] < 0.05).sum(), num, 0)
    check(
        "Selected passing combined BH", ((gh.weight != 0) & (gh.fdr_combined < 0.05)).sum(), 262, 0
    )
    check("Temporal Spearman", gh.log2fc_6h.corr(gh.log2fc_24h, method="spearman"), 0.817, 0.0005)
    check("Cells", len(meta), 6836, 0)
    check("Mapped signature genes", len(selected), 153, 0)
    check("Nonconstant genes", len(ids), 150, 0)
    check("Inside interval", inside.sum(), 1833, 0)
    check("Original peak", win["peak"], 0.718, 0.0005)
    check("Original lower edge", win["window_lower"], 0.515, 0.0005)
    check("Original upper edge", win["window_upper"], 0.793, 0.0005)
    check("Cartilage-cluster cells", len(tk), 5457, 0)
    check("Restricted peak", sk["peak"], 0.017, 0.0005)
    check("Interior amplitude", metrics(curve(ids, w))[1], 0.70, 0.005)
    check("Three-variable matched interior p", revision_null_p[1, 1], 0.141, 0.0005)
    for key, n in [
        ("weight_permutation_p_amplitude", 0.0719),
        ("matched_gene_p_amplitude", 0.0869),
        ("weight_permutation_p_variance", 0.038),
        ("matched_gene_p_variance", 0.036),
    ]:
        check(key, h1["summary"][key], n, 0.0005)
    for acc, expect in [
        ("GSE87605", (0.0186, 0.177, 0.202)),
        ("GSE14007", (0.0817, 1 / 1001, 1 / 1001)),
    ]:
        for k, v0 in zip(
            ["observed_structure_statistic", "label_permutation_p", "matched_gene_p"], expect
        ):
            check(acc + " " + k, h2[acc][k], v0, 0.0005)
    check("Shared symbols", len(cons), 14499, 0)
    check("Paired similarity", cons.conservation.mean(), 0.84047, 0.000005)
    check("Shuffled mean", null.mean(), 0.83637, 0.000005)
    check("Ortholog pairs", len(orth_sim), 13680, 0)
    check("Ortholog similarity", orth_sim.mean(), 0.84054845, 0.00000001)
    check("Signature subset n", used.sum(), 144, 0)
    check("Signature subset shuffle p", (1 + (draws >= obs).sum()) / 10001, 0.0109, 0.00005)
    for name, df, target, tol in [
        ("Size", size_sens, 0.00323, 0.000005),
        ("Weight", weight_sens, 0.133, 0.0005),
        ("Graph", root_sens, 0.0905, 0.00005),
    ]:
        check(name + " peak range", np.ptp(df.peak), target, tol)
    for method, rank in [("primary", 1), ("effect_only", 5), ("evidence_only", 110)]:
        check("Wnt5a " + method, rank_matrix.loc["WNT5A", method], rank, 0)
    alternatives = rank_summary[rank_summary.weight_method != "primary"]
    check(
        "Candidate rank correlation", alternatives.rank_spearman_vs_primary.median(), 0.442, 0.0005
    )
    check("Top-20 Jaccard", alternatives.top_k_jaccard_vs_primary.median(), 0.16, 0.005)
    check("Noise positives", hits[0], 17, 0)
    check("Signal positives", hits[1], 200, 0)
    audit = pd.DataFrame(checks)
    audit.to_csv(out / "reproduction_audit.csv", index=False)
    (root / "environment.json").write_text(
        json.dumps({"python": sys.version, "versions": versions}, indent=2)
    )
    display(audit)
    assert audit.passed.all(), audit.loc[~audit.passed].to_string(index=False)
    print(f"{len(audit)} checks passed. Results: {display_path(root)}")

    mgi_digest = "966575a12e72f61c7d5a23f3844ee2f2f11d1fa22ef044c0ec58597de4d9a05f"
    mgi_bytes = gzip.decompress((annotation_dir / "mgi_gpl922.tsv.gz").read_bytes())
    assert hashlib.sha256(mgi_bytes).hexdigest() == mgi_digest
    mgi = pd.read_csv(io.BytesIO(mgi_bytes), sep="\t")
    print(len(mgi), "source-label to approved-symbol records")

    array_marker_audit(inputs, gh, mgi, out)
    age_results = age_and_control_analysis(inputs, out, gh, mgi)
    display(
        age_results.loc[
            age_results.atlas.isin(["E15", "P28"]),
            [
                "atlas",
                "signature",
                "n_active",
                "amplitude",
                "matched_p",
                "late_minus_early",
                "direction_holm",
            ],
        ]
    )

    validate_postnatal_formats(inputs, out)
    anatomical_transfer(inputs, out)
    marker_composition_test(out)
    display(pd.read_csv(out / "P9_control_transfer.csv"))
    display(pd.read_csv(out / "marker_composition_test.csv"))

    analysis_figures(out)
    fig_response_window_sensitivity(pd.read_csv(out / "retained_size.csv"))
    fig_signature_weighting_sensitivity(pd.read_csv(out / "retained_weights.csv"))
    fig_pseudotime_root_knn_robustness(pd.read_csv(out / "retained_graph.csv"))
    publication_tables(out, h1)
    display(pd.read_csv(out / "table1.csv"))
    display(pd.read_csv(out / "H1_figure_values.csv"))

    verify_revision(out, figdir, root)
    display(pd.read_csv(out / "revision_audit.csv"))
    print("Complete analysis in", display_path(root))
