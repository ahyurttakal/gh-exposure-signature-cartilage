"""Microarray processing, sparse expression graphs and historical permutation tests."""

import os
from pathlib import Path
import re, gzip, io, math, warnings, json, hashlib, shutil, time, urllib.request, tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import mmread
from scipy.stats import norm, ttest_1samp, fisher_exact, spearmanr, rankdata
from scipy.interpolate import UnivariateSpline
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from threadpoolctl import threadpool_limits

from PIL import Image as PILImage

thread_limits = threadpool_limits(limits=2)


def export_panel(figure, name):
    """Keep a native PDF plus two visually equivalent 300-dpi image exports."""
    figdir = Path(os.environ["GHIR_FIGURES"])
    pdf, png, eps = [figdir / f"{name}.{ext}" for ext in ("pdf", "png", "eps")]
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    with PILImage.open(png) as source:
        rgb = np.asarray(source.convert("RGB"))
    height, width = rgb.shape[:2]
    plate = plt.figure(figsize=(width / 300, height / 300), dpi=300)
    plate.figimage(rgb, origin="upper", resize=False)
    plate.savefig(eps, format="eps", dpi=300)
    plt.close(plate)
    plt.close(figure)


EPS = 1e-8

CANONICAL_GH_IGF = {
    "GHR",
    "JAK2",
    "STAT5A",
    "STAT5B",
    "CISH",
    "SOCS2",
    "IGF1",
    "IGF1R",
    "IRS1",
    "IRS2",
    "AKT1",
    "MTOR",
}

MARKER_MODULES = {
    "resting": ["Apoe", "Pthlh", "Sfrp5", "Fgfr3", "Sox9", "Acan"],
    "proliferative": ["Mki67", "Pcna", "Top2a", "Ccnb1", "Cdk1", "Ccnd1"],
    "prehypertrophic": ["Ihh", "Pth1r", "Serinc5", "Mef2c", "Runx2"],
    "hypertrophic": ["Col10a1", "Mmp13", "Alpl", "Ibsp", "Spp1"],
    "chondrocyte": ["Sox9", "Col2a1", "Acan", "Comp", "Col9a1"],
    "osteoblast": ["Sp7", "Bglap", "Col1a1", "Alpl", "Runx2"],
    "gh_transduction": ["Ghr", "Jak2", "Stat5a", "Stat5b", "Cish", "Socs2"],
    "igf_axis": ["Igf1", "Igf1r", "Irs1", "Irs2", "Akt1", "Mtor"],
}

GSE3689_PUBLISHED_TARGETS = {
    "IGF1": {"6h": 2.60, "24h": 2.29},
    "TBX3": {"6h": 2.00, "24h": 2.50},
    "WNT5A": {"6h": 1.87, "24h": 2.03},
    "BMP2": {"6h": 0.74, "24h": 0.78},
    "BMP7": {"24h": 0.51},
    "STAT5A": {"24h": 0.60},
}

CONSERVATION_COLUMNS = [
    "conservation",
    "mouse_resting",
    "mouse_proliferative",
    "mouse_hypertrophic",
    "human_resting",
    "human_proliferative",
    "human_hypertrophic",
]


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    if multipletests is not None:
        return multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]
    n = len(p)
    order = np.argsort(np.nan_to_num(p, nan=1.0))
    q = np.empty(n, float)
    ranked = np.nan_to_num(p, nan=1.0)[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def percentile_score(x: pd.Series, higher_better: bool = True) -> pd.Series:
    arr = x.astype(float)
    ok = arr.notna()
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if ok.sum() == 0:
        return out
    ranks = rankdata(arr[ok].to_numpy(), method="average") / ok.sum()
    if not higher_better:
        ranks = 1.0 - ranks + 1.0 / ok.sum()
    out.loc[ok] = ranks
    return out


def _clean_soft_value(v: str) -> str:
    return v.strip().strip('"')


def parse_geo_family_soft(path: os.PathLike | str) -> Dict[str, object]:
    """Parse GEO family SOFT without GEOparse.

    Returns {samples: {GSM: {metadata, table}}, platforms: {GPL: {metadata, table}}}.
    This parser is intentionally limited to fields needed by this project.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    result = {"samples": {}, "platforms": {}}
    entity_type = None
    entity_id = None
    metadata: Dict[str, List[str]] = {}
    table_lines: List[str] = []
    in_table = False

    def flush():
        nonlocal entity_type, entity_id, metadata, table_lines, in_table
        if entity_type and entity_id:
            table = None
            if table_lines:
                try:
                    table = pd.read_csv(
                        io.StringIO("\n".join(table_lines)), sep="\t", dtype=str, low_memory=False
                    )
                except Exception as e:
                    warnings.warn(f"Could not parse table for {entity_id}: {e}")
            target = result["samples"] if entity_type == "SAMPLE" else result["platforms"]
            target[entity_id] = {"metadata": metadata.copy(), "table": table}
        metadata = {}
        table_lines = []
        in_table = False

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if line.startswith("^SAMPLE = "):
                flush()
                entity_type, entity_id = "SAMPLE", line.split("=", 1)[1].strip()
                continue
            if line.startswith("^PLATFORM = "):
                flush()
                entity_type, entity_id = "PLATFORM", line.split("=", 1)[1].strip()
                continue
            if entity_type is None:
                continue
            low = line.lower()
            if low in {"!sample_table_begin", "!platform_table_begin"}:
                in_table = True
                table_lines = []
                continue
            if low in {"!sample_table_end", "!platform_table_end"}:
                in_table = False
                continue
            if in_table:
                if line and not line.startswith("!"):
                    table_lines.append(line)
                continue
            if line.startswith("!") and " = " in line:
                k, v = line[1:].split(" = ", 1)
                metadata.setdefault(k, []).append(_clean_soft_value(v))
    flush()
    return result


def metadata_text(meta: Mapping[str, Sequence[str]]) -> str:
    pieces = []
    for k, v in meta.items():
        if isinstance(v, (list, tuple)):
            pieces.extend(map(str, v))
        else:
            pieces.append(str(v))
    return " ".join(pieces)


def choose_gene_column(platform: pd.DataFrame) -> str:
    candidates = []
    for c in platform.columns:
        lc = str(c).lower()
        score = 0
        if "gene symbol" in lc or "gene_symbol" in lc:
            score += 10
        if lc in {"symbol", "genesymbol", "gene"}:
            score += 8
        if "symbol" in lc:
            score += 5
        if "gene" in lc and "title" not in lc and "description" not in lc:
            score += 2
        candidates.append((score, c))
    candidates.sort(reverse=True, key=lambda z: z[0])
    if not candidates or candidates[0][0] <= 0:
        raise ValueError(
            f"No gene-symbol-like column found. Columns: {platform.columns.tolist()[:30]}"
        )
    return candidates[0][1]


def choose_id_column(platform: pd.DataFrame) -> str:
    for c in platform.columns:
        if str(c).upper() in {"ID", "ID_REF", "SPOT_ID"}:
            return c
    return platform.columns[0]


def clean_gene_symbol(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "null", "---", "na"}:
        return None
    for sep in [" /// ", "///", ";", "|", ","]:
        if sep in s:
            s = s.split(sep)[0].strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", s) is None:
        toks = re.findall(r"[A-Za-z][A-Za-z0-9_.-]+", s)
        if not toks:
            return None
        s = toks[0]
    return s


def platform_probe_to_gene(platform_table: pd.DataFrame) -> pd.Series:
    idc = choose_id_column(platform_table)
    gc = choose_gene_column(platform_table)
    ser = platform_table[[idc, gc]].copy()
    ser[idc] = ser[idc].astype(str)
    ser[gc] = ser[gc].map(clean_gene_symbol)
    ser = ser.dropna(subset=[gc]).drop_duplicates(idc)
    return ser.set_index(idc)[gc]


def extract_value_column(df: pd.DataFrame) -> str:
    candidates = ["VALUE", "signal", "Signal", "VALUE Signal"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_scores = []
    for c in df.columns:
        if str(c).upper() == "ID_REF":
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        numeric_scores.append((v.notna().mean(), c))
    numeric_scores.sort(reverse=True)
    if not numeric_scores or numeric_scores[0][0] < 0.5:
        raise ValueError("No numeric expression column detected")
    return numeric_scores[0][1]


def collapse_probe_matrix_to_genes(
    probe_matrix: pd.DataFrame, probe_to_gene: pd.Series, agg: str = "median"
) -> pd.DataFrame:
    """Map probe-level expression to a unique, uppercase gene-symbol index.

    Uppercasing is done *before* aggregation. This prevents case variants or
    platform aliases (for example, ``Socs2`` and ``SOCS2``) from creating
    duplicate labels later when mouse and human evidence streams are aligned.
    """
    x = probe_matrix.copy()
    x.index = x.index.astype(str)
    p2g = probe_to_gene.copy()
    p2g.index = p2g.index.astype(str)
    if p2g.index.has_duplicates:
        p2g = p2g[~p2g.index.duplicated(keep="first")]
    common = x.index.intersection(p2g.index)
    x = x.loc[common].copy()
    genes = pd.Series(p2g.reindex(common).values, index=x.index, dtype="object")
    genes = genes.map(clean_gene_symbol).map(lambda z: str(z).upper() if z is not None else np.nan)
    x["__gene__"] = genes.values
    x = x.dropna(subset=["__gene__"])
    x = x[x["__gene__"].astype(str).str.len() > 0]
    if agg == "mean":
        out = x.groupby("__gene__", sort=True).mean(numeric_only=True)
    else:
        out = x.groupby("__gene__", sort=True).median(numeric_only=True)
    out.index = out.index.astype(str)
    return out


def _canonical_gene_series(s: pd.Series, agg: str = "mean") -> pd.Series:
    """Return a gene-indexed Series with unique uppercase labels.

    Duplicate labels can arise after case normalization, multi-probe mapping,
    or alias collapse. They are aggregated rather than passed to ``reindex``,
    because pandas intentionally raises ``ValueError`` for reindexing a source
    axis that contains duplicates.
    """
    if s is None:
        return pd.Series(dtype=float)
    out = s.copy()
    out.index = pd.Index([str(x).strip().upper() for x in out.index])
    out = out[out.index.notna()]
    out = out[out.index.astype(str).str.len() > 0]
    if out.index.has_duplicates:
        if pd.api.types.is_numeric_dtype(out):
            out = out.groupby(level=0).mean() if agg == "mean" else out.groupby(level=0).median()
        else:
            out = out.groupby(level=0).first()
    return out


def _canonical_gene_frame(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """Return a gene-indexed DataFrame with unique uppercase labels."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.Index([str(x).strip().upper() for x in out.index])
    out = out[out.index.astype(str).str.len() > 0]
    if out.index.has_duplicates:
        num_cols = list(out.select_dtypes(include=[np.number, "bool"]).columns)
        other_cols = [c for c in out.columns if c not in num_cols]
        pieces = []
        if num_cols:
            num = (
                out[num_cols].groupby(level=0).mean()
                if agg == "mean"
                else out[num_cols].groupby(level=0).median()
            )
            pieces.append(num)
        if other_cols:
            pieces.append(out[other_cols].groupby(level=0).first())
        out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=sorted(set(out.index)))
    return out


def infer_gse3689_time(meta: Mapping[str, Sequence[str]], idx: int) -> str:
    """Infer the post-injection time for one GSE3689 two-color array.

    GSE3689 stores GH and PBS on the two channels of the same array.  Time is
    therefore an *array-level* attribute.  We prioritize treatment/title fields
    and only use the original replicate-order convention as a documented
    fallback (replicates 1--3: 6 h; 4--6: 24 h).
    """
    preferred = []
    for key, vals in meta.items():
        lk = str(key).lower()
        if any(
            tok in lk for tok in ["treatment_protocol", "title", "description", "characteristics"]
        ):
            preferred.extend(vals if isinstance(vals, (list, tuple)) else [vals])
    txt = " ".join(map(str, preferred)).lower() or metadata_text(meta).lower()
    if re.search(r"\b6\s*hours?\b|\b6\s*h\b", txt):
        return "6h"
    if re.search(r"\b24\s*hours?\b|\b24\s*h\b", txt):
        return "24h"
    return "6h" if idx < 3 else "24h"


def infer_gse3689_channel_orientation(
    meta: Mapping[str, Sequence[str]],
) -> Tuple[int, str, str, str]:
    """Return sign needed for GH/PBS log-ratio and audit strings.

    The deposited experiment labels GH-treated RNA with Cy5 on channel 1 and
    PBS control RNA with Cy3 on channel 2.  We nevertheless infer the channel
    assignment from the sample metadata whenever possible so that a silent
    channel reversal cannot flip every effect direction.
    """

    def collect(ch: int) -> str:
        pieces = []
        suffix = f"_ch{ch}"
        for key, vals in meta.items():
            lk = str(key).lower()
            if lk.endswith(suffix) and any(
                tok in lk
                for tok in ["source_name", "treatment_protocol", "label", "characteristics"]
            ):
                pieces.extend(vals if isinstance(vals, (list, tuple)) else [vals])
        return " ".join(map(str, pieces)).lower()

    ch1 = collect(1)
    ch2 = collect(2)
    gh1 = bool(re.search(r"growth hormone|\bgh\b", ch1))
    pbs1 = "pbs" in ch1
    gh2 = bool(re.search(r"growth hormone|\bgh\b", ch2))
    pbs2 = "pbs" in ch2
    if gh1 and pbs2:
        return 1, "metadata", "GH", "PBS"
    if gh2 and pbs1:
        return -1, "metadata", "PBS", "GH"
    # Dataset-design fallback is explicit and auditable, never silent.
    return 1, "GSE3689 documented design fallback", "GH", "PBS"


def _gse3689_soft_columns_shifted(df: pd.DataFrame) -> bool:
    """Detect the historical one-column shift in GSE3689 family SOFT tables.

    In the GEO family representation the header can include VALUE even when the
    row-level processed ratio is absent.  The raw channel values are then shifted
    one column to the left.  Treating the apparent CH1MedianSig/CH2MedianSig
    labels literally in that case mixes foreground and background columns and
    produces an invalid GH signature.
    """
    needed = {
        "VALUE",
        "CH1MeanSig",
        "CH2MeanSig",
        "CH1MedianSig",
        "CH2MedianSig",
        "CH1MeanBac",
        "CH2MeanBac",
        "CH1MedianBac",
        "CH2MedianBac",
    }
    if not needed.issubset(df.columns):
        return False
    value = pd.to_numeric(df["VALUE"], errors="coerce")
    pre = (
        pd.to_numeric(df["PRE_VALUE"], errors="coerce")
        if "PRE_VALUE" in df.columns
        else pd.Series(dtype=float)
    )
    finite = value[np.isfinite(value)]
    if finite.empty:
        return False
    # A true log2 ratio is normally O(1), whereas the shifted VALUE column
    # contains foreground fluorescence intensities O(10^2--10^4).
    value_looks_raw = float(np.nanmedian(np.abs(finite))) > 20.0
    pre_missing = (len(pre) == 0) or (float(np.isfinite(pre).mean()) < 0.05)
    return bool(value_looks_raw and pre_missing)


def _repair_gse3689_channel_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Return correctly named raw channel columns for a GSE3689 sample table."""
    raw = df.copy()
    shifted = _gse3689_soft_columns_shifted(raw)
    cols = [
        "CH1MeanSig",
        "CH2MeanSig",
        "CH1MedianSig",
        "CH2MedianSig",
        "CH1MeanBac",
        "CH2MeanBac",
        "CH1MedianBac",
        "CH2MedianBac",
    ]
    if shifted:
        # GEO row values omit VALUE; all raw-channel fields therefore appear
        # under the preceding header.  The mapping below restores the intended
        # Agilent Feature Extraction fields.
        mapping = {
            "CH1MeanSig": "VALUE",
            "CH2MeanSig": "CH1MeanSig",
            "CH1MedianSig": "CH2MeanSig",
            "CH2MedianSig": "CH1MedianSig",
            "CH1MeanBac": "CH2MedianSig",
            "CH2MeanBac": "CH1MeanBac",
            "CH1MedianBac": "CH2MeanBac",
            "CH2MedianBac": "CH1MedianBac",
        }
        fixed = pd.DataFrame(index=raw.index)
        for dest, src in mapping.items():
            fixed[dest] = pd.to_numeric(raw[src], errors="coerce")
    else:
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            raise ValueError(f"GSE3689 raw-channel columns missing: {missing}")
        fixed = raw[cols].apply(pd.to_numeric, errors="coerce")
    fixed.index = raw.index
    return fixed, shifted


def _gse3689_lowess_logratio(
    df: pd.DataFrame,
    meta: Mapping[str, Sequence[str]],
    raw_signal_threshold: float = 100.0,
    lowess_frac: float = 0.30,
) -> Tuple[pd.Series, Dict[str, object]]:
    """Reconstruct one GSE3689 array using the published preprocessing logic.

    Govoni et al. analyzed local-background-subtracted median channel signals,
    applied within-array LOWESS normalization, and retained probes with raw
    signal >=100 and a present flag.  The family SOFT file does not expose the
    original GeneSpring present/absent flag, so this reanalysis reproduces the
    available parts exactly and records that flag filtering is unavailable.
    """
    if sm_lowess is None:
        raise RuntimeError(
            "statsmodels LOWESS is required for GSE3689 preprocessing. Install statsmodels>=0.14."
        )
    fixed, shifted = _repair_gse3689_channel_columns(df)
    ch1_med = fixed["CH1MedianSig"].to_numpy(float)
    ch2_med = fixed["CH2MedianSig"].to_numpy(float)
    ch1_bac = fixed["CH1MedianBac"].to_numpy(float)
    ch2_bac = fixed["CH2MedianBac"].to_numpy(float)
    fg1 = ch1_med - ch1_bac
    fg2 = ch2_med - ch2_bac
    rawmax = np.fmax(ch1_med, ch2_med)
    valid = (
        np.isfinite(fg1)
        & np.isfinite(fg2)
        & np.isfinite(rawmax)
        & (fg1 > 0)
        & (fg2 > 0)
        & (rawmax >= float(raw_signal_threshold))
    )
    M = np.full(len(df), np.nan, float)
    A = np.full(len(df), np.nan, float)
    M[valid] = np.log2(fg1[valid] / fg2[valid])
    A[valid] = 0.5 * (np.log2(fg1[valid]) + np.log2(fg2[valid]))
    if valid.sum() < 100:
        raise ValueError(
            f"Only {int(valid.sum())} GSE3689 probes passed signal/background QC; cannot fit LOWESS reliably."
        )
    fitted = sm_lowess(M[valid], A[valid], frac=float(lowess_frac), it=3, return_sorted=False)
    normM = np.full(len(df), np.nan, float)
    normM[valid] = M[valid] - np.asarray(fitted, float)
    sign, orientation_source, ch1_treat, ch2_treat = infer_gse3689_channel_orientation(meta)
    normM *= float(sign)
    out = pd.Series(normM, index=df["ID_REF"].astype(str), dtype=float)
    qc = {
        "soft_column_shift_repaired": bool(shifted),
        "channel_orientation_source": orientation_source,
        "channel1_treatment": ch1_treat,
        "channel2_treatment": ch2_treat,
        "n_spots_total": int(len(df)),
        "n_spots_signal_pass": int(valid.sum()),
        "signal_pass_fraction": float(valid.mean()),
        "raw_signal_threshold": float(raw_signal_threshold),
        "lowess_fraction": float(lowess_frac),
        "median_raw_M_before_lowess": float(np.nanmedian(M[valid])),
        "median_M_after_lowess": float(np.nanmedian(normM[valid])),
        "present_flag_available": False,
        "preprocessing_method": "background-subtracted median + robust LOWESS",
    }
    return out, qc


def construct_gh_weights(
    gh_stats: pd.DataFrame,
    method: str = "primary",
    top_n: int = 300,
    winsor_quantile: float = 0.995,
) -> pd.DataFrame:
    """Construct normalized acute-GH weights from one common GSE3689 statistic table.

    Candidate selection and single-cell projection must use the *same* primary
    perturbational score.  ``selected_signature`` therefore explicitly records
    the genes entering the top-N primary/alternative signature and is used by
    the candidate-ranking stage rather than ranking every assayed GSE3689 gene.
    """
    x = _canonical_gene_frame(gh_stats.copy())
    required = {"mean_log2fc", "z_combined"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(
            f"GH statistics missing required columns for weight construction: {sorted(missing)}"
        )
    mu = pd.to_numeric(x["mean_log2fc"], errors="coerce")
    z = pd.to_numeric(x["z_combined"], errors="coerce")
    method = str(method).lower()
    if method == "primary":
        raw = np.sign(mu) * np.abs(z) * np.sqrt(np.abs(mu) + 0.05)
    elif method == "effect_only":
        raw = mu.copy()
    elif method == "evidence_only":
        raw = np.sign(mu) * np.abs(z)
    else:
        raise ValueError("method must be one of: primary, effect_only, evidence_only")
    eligible = pd.Series(True, index=x.index)
    if "analysis_eligible" in x.columns:
        eligible = x["analysis_eligible"].fillna(False).astype(bool)
    raw = raw.where(eligible & np.isfinite(raw), np.nan)
    finite = np.isfinite(raw.to_numpy(float))
    lim = (
        float(np.nanquantile(np.abs(raw.to_numpy(float)[finite]), winsor_quantile))
        if finite.any()
        else 1.0
    )
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    raw_w = raw.clip(-lim, lim)
    x["weight_raw_unwinsorized"] = raw
    x["weight_raw"] = raw_w
    x["rank_abs_weight"] = np.nan
    if raw_w.notna().any():
        x.loc[raw_w.notna(), "rank_abs_weight"] = (
            raw_w.dropna().abs().rank(ascending=False, method="min")
        )
    selected_idx = raw_w.dropna().abs().nlargest(min(int(top_n), int(raw_w.notna().sum()))).index
    x["selected_signature"] = x.index.isin(selected_idx)
    x["weight"] = 0.0
    if len(selected_idx):
        norm_l2 = float(np.sqrt(np.sum(raw_w.loc[selected_idx].to_numpy(float) ** 2))) or 1.0
        x.loc[selected_idx, "weight"] = raw_w.loc[selected_idx] / norm_l2
    x["weight_method"] = method
    return x.sort_values(["selected_signature", "rank_abs_weight"], ascending=[False, True])


def analyze_gse3689(
    geo: Dict[str, object],
    top_n: int = 300,
    raw_signal_threshold: float = 100.0,
    lowess_frac: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reanalyze GSE3689 with channel-aware, background-subtracted LOWESS preprocessing.

    This function deliberately does not compute log2(CH1MedianSig/CH2MedianSig)
    from the literal family-SOFT headers because the historical GSE3689 SOFT
    representation can be one column shifted.  Each array is repaired,
    background-subtracted, LOWESS normalized, and oriented as GH/PBS before
    probe-to-gene collapse.
    """
    platforms = geo["platforms"]
    if not platforms:
        raise ValueError("GSE3689 platform table missing")
    platform_id = list(platforms.keys())[0]
    p2g = platform_probe_to_gene(platforms[platform_id]["table"])
    sample_series = []
    sample_meta = []
    for idx, (gsm, obj) in enumerate(geo["samples"].items()):
        df = obj["table"]
        if df is None:
            continue
        if "ID_REF" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ID_REF"})
        series, qc = _gse3689_lowess_logratio(
            df, obj["metadata"], raw_signal_threshold, lowess_frac
        )
        series.name = gsm
        sample_series.append(series)
        qc.update({"sample": gsm, "time": infer_gse3689_time(obj["metadata"], idx)})
        sample_meta.append(qc)
    if not sample_series:
        raise ValueError("No usable GSE3689 sample tables")
    probe = pd.concat(sample_series, axis=1)
    gene = collapse_probe_matrix_to_genes(probe, p2g, agg="median")
    sm = pd.DataFrame(sample_meta).set_index("sample")
    rows = []
    for g, row in gene.iterrows():
        rec = {"gene": g}
        zs = []
        effects = []
        available_z = []
        for t in ["6h", "24h"]:
            cols = sm.index[sm["time"] == t].intersection(gene.columns)
            vals = pd.to_numeric(row[cols], errors="coerce").dropna().to_numpy(float)
            rec[f"n_valid_{t}"] = int(len(vals))
            mu = float(np.mean(vals)) if len(vals) else np.nan
            if len(vals) >= 2 and np.nanstd(vals, ddof=1) > 0:
                _, p = ttest_1samp(vals, 0.0, nan_policy="omit")
                p = float(p)
            else:
                p = 1.0
            z = (
                float(np.sign(mu) * norm.ppf(1 - max(min(p, 1 - 1e-15), 1e-300) / 2))
                if np.isfinite(mu)
                else np.nan
            )
            rec[f"log2fc_{t}"] = mu
            rec[f"p_{t}"] = p
            rec[f"z_{t}"] = z
            if np.isfinite(z):
                available_z.append(z)
            if np.isfinite(mu):
                effects.append(mu)
        rec["n_timepoints_available"] = int(len(available_z))
        rec["z_combined"] = (
            float(np.sum(available_z) / math.sqrt(len(available_z))) if available_z else np.nan
        )
        rec["mean_log2fc"] = float(np.mean(effects)) if effects else np.nan
        # At least one time point must have >=2 valid paired arrays.
        rec["analysis_eligible"] = bool(rec["n_valid_6h"] >= 2 or rec["n_valid_24h"] >= 2)
        rows.append(rec)
    res = pd.DataFrame(rows).set_index("gene")
    for t in ["6h", "24h"]:
        res[f"fdr_{t}"] = bh_fdr(res[f"p_{t}"].fillna(1.0).values)
    res["p_combined"] = 2 * norm.sf(
        np.abs(pd.to_numeric(res["z_combined"], errors="coerce").fillna(0.0))
    )
    res.loc[~res["analysis_eligible"], "p_combined"] = 1.0
    res["fdr_combined"] = bh_fdr(res["p_combined"].values)
    res = construct_gh_weights(res, method="primary", top_n=top_n)
    return res, sm.reset_index()


def gse3689_published_target_qc(gh: pd.DataFrame) -> pd.DataFrame:
    """Compare reanalysis directions with fold changes explicitly reported by Govoni et al."""
    g = _canonical_gene_frame(gh.copy())
    rows = []
    for gene, vals in GSE3689_PUBLISHED_TARGETS.items():
        for time, pub_fold in vals.items():
            col = f"log2fc_{time}"
            est = (
                float(g.loc[gene, col])
                if gene in g.index and col in g.columns and np.isfinite(g.loc[gene, col])
                else np.nan
            )
            pub_log2 = float(np.log2(pub_fold))
            rows.append(
                {
                    "gene": gene,
                    "time": time,
                    "published_fold_change": float(pub_fold),
                    "published_log2fc": pub_log2,
                    "estimated_log2fc": est,
                    "estimated_fold_change": float(2**est) if np.isfinite(est) else np.nan,
                    "direction_match": bool(np.sign(est) == np.sign(pub_log2))
                    if np.isfinite(est)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def infer_zone(text: str, human: bool = False) -> Optional[str]:
    """Infer one growth-plate zone from a *sample-specific* text string.

    Important: callers should not pass the concatenation of the entire GEO
    metadata record. GEO extraction protocols commonly describe *all* growth
    plate zones, which can cause every sample to be assigned to the same zone.
    ``infer_zone_from_metadata`` below therefore prioritizes the sample title.
    """
    s = str(text).strip().lower()
    if not s:
        return None
    # Most specific labels must be checked before their parent terms.
    if human:
        if "prehypert" in s or re.search(r"\bphz\d*\b", s):
            return "prehypertrophic"
        if "hypertroph" in s or re.search(r"\bhz\d*\b", s):
            return "hypertrophic"
        if "prolifer" in s or re.search(r"\bpz\d*\b", s):
            return "proliferative"
        if "reserve" in s or "rest" in s or "round" in s or re.search(r"\brz\d*\b", s):
            return "resting"
        if "perichond" in s or re.search(r"\bpc\b", s):
            return "perichondrium"
    else:
        if "prehypert" in s or re.search(r"\bphz\d*\b", s):
            return "prehypertrophic"
        if "hypertroph" in s or re.search(r"\bhz\d*\b", s):
            return "hypertrophic"
        if "matur" in s or re.search(r"\bmz\d*\b", s):
            return "maturing"
        if "prolifer" in s or "flat" in s or re.search(r"\bpz\d*\b", s):
            return "proliferative"
        if "round" in s or "rest" in s or re.search(r"\brz\d*\b", s):
            return "resting"
    return None


def _metadata_values(
    meta: Mapping[str, Sequence[str]], key_fragments: Sequence[str]
) -> List[Tuple[str, str]]:
    """Return metadata values whose key matches one of ``key_fragments``."""
    out = []
    for k, v in meta.items():
        lk = str(k).lower()
        if not any(fragment in lk for fragment in key_fragments):
            continue
        values = v if isinstance(v, (list, tuple)) else [v]
        for item in values:
            out.append((str(k), str(item)))
    return out


def infer_zone_from_metadata(
    meta: Mapping[str, Sequence[str]], human: bool = False
) -> Tuple[Optional[str], str, str]:
    """Infer zone using sample-specific GEO fields in a strict priority order.

    The v1.2 implementation concatenated all sample metadata, including common
    extraction protocols that mention every growth-plate zone. For GSE9160 this
    could label all ten samples as the same zone; for GSE14007 it could similarly
    collapse the four-zone replication to one zone. This function fixes that by
    using title/source/characteristics before any broad fallback.

    Returns ``(zone, source_field, source_text)`` for diagnostics.
    """
    priorities = [
        ("title", ["sample_title", "title"]),
        ("source", ["source_name_ch1", "source name"]),
        ("characteristics", ["characteristics_ch1", "characteristics"]),
        ("description", ["description"]),
    ]
    for label, frags in priorities:
        for key, value in _metadata_values(meta, frags):
            zone = infer_zone(value, human=human)
            if zone is not None:
                return zone, key, value
    # Last-resort fallback excludes protocol/method fields because these often
    # mention all zones and are not sample-identifying.
    excluded = ("protocol", "extract", "hybrid", "scan", "data_processing", "growth")
    for k, v in meta.items():
        lk = str(k).lower()
        if any(tok in lk for tok in excluded):
            continue
        values = v if isinstance(v, (list, tuple)) else [v]
        for item in values:
            zone = infer_zone(str(item), human=human)
            if zone is not None:
                return zone, str(k), str(item)
    return None, "", ""


def process_zonal_geo(
    geo: Dict[str, object], human: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    platforms = geo["platforms"]
    if not platforms:
        raise ValueError("Platform table missing")
    platform_id = list(platforms.keys())[0]
    p2g = platform_probe_to_gene(platforms[platform_id]["table"])
    ss = []
    meta = []
    for gsm, obj in geo["samples"].items():
        df = obj["table"]
        if df is None:
            continue
        if "ID_REF" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ID_REF"})
        vc = extract_value_column(df)
        vals = pd.to_numeric(df[vc], errors="coerce")
        s = pd.Series(vals.values, index=df["ID_REF"].astype(str), name=gsm)
        ss.append(s)
        txt = metadata_text(obj["metadata"])
        zone, zone_source, zone_text = infer_zone_from_metadata(obj["metadata"], human=human)
        meta.append(
            {
                "sample": gsm,
                "zone": zone,
                "zone_source": zone_source,
                "zone_text": zone_text[:160],
                "text": txt[:300],
            }
        )
    probe = pd.concat(ss, axis=1)
    # scale each sample to common median to reduce arbitrary between-array scaling.
    med = probe.median(axis=0, skipna=True)
    target = np.nanmedian(med.values)
    probe = probe.multiply(target / (med + EPS), axis=1)
    gene = collapse_probe_matrix_to_genes(probe, p2g, agg="median")
    m = pd.DataFrame(meta).set_index("sample")
    valid = m["zone"].notna()
    m = m.loc[valid]
    gene = gene[m.index.intersection(gene.columns)]
    zone_means = {}
    for z in sorted(m["zone"].unique()):
        cols = m.index[m["zone"] == z].intersection(gene.columns)
        zone_means[z] = gene[cols].mean(axis=1)
    return pd.DataFrame(zone_means), m.reset_index()


def zone_specificity(
    zone_mean: pd.DataFrame, zones: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    x = zone_mean.copy().astype(float).clip(lower=0)
    if zones is not None:
        keep = [z for z in zones if z in x.columns]
        missing = [z for z in zones if z not in x.columns]
        if missing:
            warnings.warn(
                f"Requested growth-plate zones are missing: {missing}; "
                f"available zones={list(x.columns)}. Check sample-zone parsing."
            )
        x = x[keep]
    denom = x.sum(axis=1) + EPS
    return x.div(denom, axis=0)


def zonal_gh_propensity(spec: pd.DataFrame, gh: pd.DataFrame) -> pd.DataFrame:
    w = _canonical_gene_series(gh["weight"].copy())
    w = w.loc[pd.to_numeric(w, errors="coerce").fillna(0).abs() > 0]
    s = _canonical_gene_frame(spec.copy())
    common = s.index.intersection(w.index)
    if len(common) == 0:
        raise ValueError("No selected GH-signature gene overlap for zonal GH propensity")
    wv = w.loc[common]
    den = float(np.abs(wv).sum()) or 1.0
    rows = []
    for z in s.columns:
        pi = s.loc[common, z]
        rows.append(
            {
                "zone": z,
                "signed_propensity": float((wv * pi).sum() / den),
                "unsigned_burden": float((np.abs(wv) * pi).sum() / den),
                "n_genes": len(common),
            }
        )
    return pd.DataFrame(rows).set_index("zone")


def _find_file(directory: Path, patterns: Sequence[str]) -> Path:
    files = list(directory.rglob("*"))
    for p in patterns:
        reg = re.compile(p, re.I)
        for f in files:
            if f.is_file() and reg.search(f.name):
                return f
    raise FileNotFoundError(f"No file matching {patterns} in {directory}")


def _open_text_auto(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def load_10x_mtx(directory: os.PathLike | str) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    d = Path(directory)
    mtx = _find_file(d, [r"matrix.*\.mtx(\.gz)?$", r"\.mtx(\.gz)?$"])
    feat = _find_file(d, [r"features.*\.tsv(\.gz)?$", r"genes.*\.tsv(\.gz)?$"])
    bar = _find_file(d, [r"barcodes.*\.tsv(\.gz)?$"])
    if mtx.suffix == ".gz":
        with gzip.open(mtx, "rb") as fh:
            mat = mmread(fh).tocsr()
    else:
        mat = mmread(mtx).tocsr()
    # 10x is genes x cells
    with _open_text_auto(feat) as fh:
        feats = [line.rstrip("\n").split("\t") for line in fh if line.strip()]
    genes = [r[1] if len(r) > 1 else r[0] for r in feats]
    with _open_text_auto(bar) as fh:
        barcodes = [line.strip().split("\t")[0] for line in fh if line.strip()]
    if mat.shape[0] == len(genes):
        X = mat.T.tocsr()
    elif mat.shape[1] == len(genes):
        X = mat.tocsr()
    else:
        raise ValueError(f"Matrix dimensions {mat.shape} incompatible with {len(genes)} genes")
    return X, genes, barcodes


def qc_and_normalize_sc(
    X: sparse.csr_matrix,
    genes: Sequence[str],
    barcodes: Sequence[str],
    min_genes: int = 200,
    max_mt_pct: float = 20.0,
    max_cells: Optional[int] = 20000,
    random_state: int = 42,
) -> Tuple[sparse.csr_matrix, List[str], pd.DataFrame]:
    genes = np.asarray(genes, dtype=object)
    total = np.asarray(X.sum(axis=1)).ravel()
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    mt = np.array([str(g).upper().startswith("MT-") for g in genes])
    mt_counts = np.asarray(X[:, mt].sum(axis=1)).ravel() if mt.any() else np.zeros(X.shape[0])
    pct_mt = 100 * mt_counts / (total + EPS)
    max_genes = max(np.quantile(n_genes, 0.995), min_genes + 100)
    keep = (n_genes >= min_genes) & (n_genes <= max_genes) & (pct_mt <= max_mt_pct) & (total > 0)
    idx = np.where(keep)[0]
    if max_cells is not None and len(idx) > max_cells:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(idx, size=max_cells, replace=False))
    X = X[idx].tocsr().astype(np.float64)
    meta = pd.DataFrame(
        {
            "barcode": np.asarray(barcodes, dtype=object)[idx],
            "total_counts": total[idx],
            "n_genes": n_genes[idx],
            "pct_mt": pct_mt[idx],
        }
    )
    sf = 1e4 / (np.asarray(X.sum(axis=1)).ravel() + EPS)
    X = X.multiply(sf[:, None]).tocsr()
    X.data = np.log1p(X.data)
    return X, list(genes), meta


def select_hvg_and_pca(
    X: sparse.csr_matrix,
    genes: Sequence[str],
    n_hvg: int = 2500,
    n_pcs: int = 30,
    random_state: int = 42,
):
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean2 = np.asarray(X.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean2 - mean**2, 0)
    dispersion = var / (mean + 0.1)
    expressed = np.asarray((X > 0).sum(axis=0)).ravel() >= 10
    score = np.where(expressed, dispersion, -np.inf)
    n = min(n_hvg, np.isfinite(score).sum())
    hvg_idx = np.argsort(score)[-n:]
    svd = TruncatedSVD(n_components=min(n_pcs, n - 1, X.shape[0] - 1), random_state=random_state)
    pcs = svd.fit_transform(X[:, hvg_idx])
    return pcs, hvg_idx, svd


def gene_lookup(genes: Sequence[str]) -> Dict[str, int]:
    out = {}
    for i, g in enumerate(genes):
        key = str(g).split("_")[0].upper()
        out.setdefault(key, i)
    return out


def module_score(
    X: sparse.csr_matrix, genes: Sequence[str], marker_genes: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    lookup = gene_lookup(genes)
    idx = []
    used = []
    for g in marker_genes:
        k = g.upper()
        if k in lookup:
            idx.append(lookup[k])
            used.append(g)
    if not idx:
        return np.zeros(X.shape[0]), []
    arr = np.asarray(X[:, idx].mean(axis=1)).ravel()
    sd = np.std(arr)
    return (arr - np.mean(arr)) / (sd + EPS), used


def annotate_states(
    X: sparse.csr_matrix, genes: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    scores = {}
    used = {}
    for name, markers in MARKER_MODULES.items():
        scores[name], used[name] = module_score(X, genes, markers)
    state_names = ["resting", "proliferative", "prehypertrophic", "hypertrophic"]
    state_mat = np.column_stack([scores[s] for s in state_names])
    state = np.asarray(state_names, dtype=object)[np.argmax(state_mat, axis=1)]
    df = pd.DataFrame({f"score_{k}": v for k, v in scores.items()})
    df["cell_state"] = state
    return df, used


def build_knn_graph(pcs: np.ndarray, n_neighbors: int = 15) -> sparse.csr_matrix:
    k = min(n_neighbors + 1, len(pcs))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(pcs)
    dist, ind = nbrs.kneighbors(pcs)
    rows = []
    cols = []
    dat = []
    for i in range(len(pcs)):
        for j, dv in zip(ind[i, 1:], dist[i, 1:]):
            rows.extend([i, j])
            cols.extend([j, i])
            dat.extend([dv, dv])
    graph = sparse.coo_matrix((dat, (rows, cols)), shape=(len(pcs), len(pcs))).tocsr()
    graph.sum_duplicates()
    return graph


def select_pseudotime_root(
    pcs: np.ndarray, state_scores: pd.DataFrame, strategy: str = "resting_minus_hypertrophic_max"
) -> int:
    """Select a biologically resting-like root using one of three predefined strategies."""
    strategy = str(strategy).lower()
    rest = pd.to_numeric(state_scores["score_resting"], errors="coerce").to_numpy(float)
    hyp = pd.to_numeric(state_scores["score_hypertrophic"], errors="coerce").to_numpy(float)
    if strategy == "resting_minus_hypertrophic_max":
        signal = rest - hyp
        return int(np.nanargmax(signal))
    if strategy == "resting_score_max":
        return int(np.nanargmax(rest))
    if strategy == "resting_centroid_medoid":
        finite = np.isfinite(rest)
        if finite.sum() < 5:
            return int(np.nanargmax(rest - hyp))
        cutoff = float(np.nanquantile(rest[finite], 0.90))
        cand = np.where(finite & (rest >= cutoff))[0]
        if len(cand) < 3:
            cand = np.argsort(np.nan_to_num(rest, nan=-np.inf))[-max(3, min(20, len(rest))) :]
        centroid = np.nanmean(pcs[cand], axis=0)
        dist = np.linalg.norm(pcs[cand] - centroid, axis=1)
        return int(cand[int(np.nanargmin(dist))])
    raise ValueError("Unknown root strategy: " + strategy)


def graph_geodesic_pseudotime(
    pcs: np.ndarray,
    state_scores: pd.DataFrame,
    n_neighbors: int = 15,
    root_strategy: str = "resting_minus_hypertrophic_max",
) -> Tuple[np.ndarray, sparse.csr_matrix, int]:
    graph = build_knn_graph(pcs, n_neighbors=n_neighbors)
    root = select_pseudotime_root(pcs, state_scores, root_strategy)
    dist = dijkstra(graph, directed=False, indices=root)
    finite = np.isfinite(dist)
    if finite.sum() < 0.8 * len(dist):
        # Restrict root selection to the largest connected component rather than silently
        # changing the biological root definition.
        ncomp, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        lab = int(np.argmax(sizes))
        candidates = np.where(labels == lab)[0]
        local = select_pseudotime_root(
            pcs[candidates], state_scores.iloc[candidates].reset_index(drop=True), root_strategy
        )
        root = int(candidates[local])
        dist = dijkstra(graph, directed=False, indices=root)
        finite = np.isfinite(dist)
    maxd = np.nanmax(dist[finite]) if finite.any() else 1.0
    pt = np.full(len(dist), np.nan)
    pt[finite] = dist[finite] / (maxd + EPS)
    return pt, graph, root


def embedding_2d(pcs, random_state=42):
    """Revision figures use the first two SVD coordinates."""
    return pcs[:, :2], "SVD"


def weighted_gh_score(
    X: sparse.csr_matrix, genes: Sequence[str], gh: pd.DataFrame, max_genes: int = 300
) -> Tuple[np.ndarray, List[str]]:
    lookup = gene_lookup(genes)
    gh2 = _canonical_gene_frame(gh.copy())
    gh2["absw"] = gh2["weight"].abs()
    gh2 = gh2.sort_values("absw", ascending=False)
    idx = []
    weights = []
    used = []
    for g, row in gh2.iterrows():
        if len(idx) >= max_genes:
            break
        k = str(g).upper()
        if k in lookup and abs(float(row["weight"])) > 0:
            idx.append(lookup[k])
            weights.append(float(row["weight"]))
            used.append(str(g))
    if len(idx) < 5:
        raise ValueError(f"Only {len(idx)} GH signature genes found in single-cell matrix")
    A = X[:, idx].toarray().astype(float)
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + EPS)
    w = np.asarray(weights, float)
    score = A @ w / (np.sqrt(np.sum(w * w)) + EPS)
    return score, used


def analyze_single_cell(
    raw_dir: os.PathLike | str,
    gh: pd.DataFrame,
    max_cells: Optional[int] = 20000,
    random_state: int = 42,
) -> Dict[str, object]:
    X, genes, barcodes = load_10x_mtx(raw_dir)
    X, genes, meta = qc_and_normalize_sc(
        X, genes, barcodes, max_cells=max_cells, random_state=random_state
    )
    state_scores, used = annotate_states(X, genes)
    pcs, hvg_idx, svd = select_hvg_and_pca(X, genes, random_state=random_state)
    pt, graph, root = graph_geodesic_pseudotime(pcs, state_scores)
    emb, emb_name = embedding_2d(pcs, random_state=random_state)
    gh_score, gh_used = weighted_gh_score(X, genes, gh)
    meta = pd.concat([meta.reset_index(drop=True), state_scores.reset_index(drop=True)], axis=1)
    meta["pseudotime"] = pt
    meta["gh_response_score"] = gh_score
    meta["embed1"] = emb[:, 0]
    meta["embed2"] = emb[:, 1]
    meta["embedding_type"] = emb_name
    return {
        "X": X,
        "genes": genes,
        "meta": meta,
        "pcs": pcs,
        "root": root,
        "used_markers": used,
        "gh_genes_used": gh_used,
    }


def smooth_binned_profile(centers, values, grid):
    """Evaluate the cubic smoothing spline for standardized bin means.

    FITPACK selects knots adaptively. Equivalent bin scores from different
    vector math implementations can differ in their last floating-point digits.
    Use a fixed precision for both observed and resampled profiles so that this
    roundoff does not decide between competing knots.
    """
    bin_scores = np.round(np.asarray(values, dtype=np.float64), decimals=9)
    return UnivariateSpline(centers, bin_scores, s=0.02 * len(centers))(grid)


def fit_response_window(
    pseudotime: Sequence[float],
    score: Sequence[float],
    n_bins: int = 30,
    threshold_fraction: float = 0.80,
    n_boot: int = 200,
    random_state: int = 42,
) -> Dict[str, object]:
    x = np.asarray(pseudotime, float)
    y = np.asarray(score, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    bidx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
    means = np.array(
        [np.mean(y[bidx == i]) if np.any(bidx == i) else np.nan for i in range(n_bins)]
    )
    valid = np.isfinite(means)
    if valid.sum() < 5:
        raise ValueError("Too few populated pseudotime bins")
    spline = UnivariateSpline(centers[valid], means[valid], s=max(0.01, valid.sum() * 0.02))
    grid = np.linspace(centers[valid].min(), centers[valid].max(), 300)
    smooth = spline(grid)
    rng = np.random.default_rng(random_state)
    boots = []
    if n_boot > 0:
        for _ in range(n_boot):
            bm = []
            bc = []
            for i in range(n_bins):
                vals = y[bidx == i]
                if len(vals) >= 2:
                    bm.append(np.mean(rng.choice(vals, size=len(vals), replace=True)))
                    bc.append(centers[i])
            if len(bm) >= 5:
                try:
                    sp = UnivariateSpline(
                        np.asarray(bc), np.asarray(bm), s=max(0.01, len(bm) * 0.02)
                    )
                    boots.append(sp(grid))
                except Exception:
                    pass
    if boots:
        B = np.vstack(boots)
        lo = np.percentile(B, 2.5, axis=0)
        hi = np.percentile(B, 97.5, axis=0)
    else:
        lo = np.full_like(grid, np.nan)
        hi = np.full_like(grid, np.nan)
    peak_idx = int(np.argmax(smooth))
    peak = float(grid[peak_idx])
    fmin = float(np.min(smooth))
    fmax = float(np.max(smooth))
    threshold = fmin + threshold_fraction * (fmax - fmin)
    above = smooth >= threshold
    # contiguous component containing peak
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(grid) - 1 and above[right + 1]:
        right += 1
    return {
        "grid": grid,
        "smooth": smooth,
        "ci_low": lo,
        "ci_high": hi,
        "peak": peak,
        "window_lower": float(grid[left]),
        "window_upper": float(grid[right]),
        "window_width": float(grid[right] - grid[left]),
        "threshold": float(threshold),
        "n_cells": len(x),
        "threshold_fraction": threshold_fraction,
    }


def _sparse_gene_moments(X: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise mean, population SD, and detection prevalence for sparse log-expression."""
    X = X.tocsr()
    mean = np.asarray(X.mean(axis=0)).ravel().astype(float)
    mean2 = np.asarray(X.power(2).mean(axis=0)).ravel().astype(float)
    var = np.maximum(mean2 - mean**2, 0.0)
    sd = np.sqrt(var)
    det = np.asarray((X != 0).mean(axis=0)).ravel().astype(float)
    return mean, sd, det


def _sparse_standardized_weighted_score(
    X: sparse.csr_matrix,
    idx: Sequence[int],
    weights: Sequence[float],
    means: np.ndarray,
    sds: np.ndarray,
) -> np.ndarray:
    """Compute the same standardized weighted score as ``weighted_gh_score`` without densifying columns."""
    idx = np.asarray(idx, dtype=int)
    w = np.asarray(weights, dtype=float)
    if len(idx) != len(w) or len(idx) < 1:
        raise ValueError("idx and weights must have equal non-zero length")
    sd = np.asarray(sds[idx], dtype=float)
    mu = np.asarray(means[idx], dtype=float)
    keep = np.isfinite(sd) & (sd > EPS) & np.isfinite(mu) & np.isfinite(w)
    idx = idx[keep]
    w = w[keep]
    sd = sd[keep]
    mu = mu[keep]
    if len(idx) < 5:
        raise ValueError("Too few non-constant genes for standardized weighted score")
    coef = w / sd
    raw = np.asarray(X[:, idx].dot(coef)).ravel() - float(np.sum(w * mu / sd))
    den = float(np.sqrt(np.sum(w * w))) + EPS
    return raw / den


def _curve_structure_stats(window: Mapping[str, object]) -> Dict[str, float]:
    """Global statistics defined for developmental structure of the smoothed response curve."""
    smooth = np.asarray(window["smooth"], dtype=float)
    return {
        "amplitude": float(np.nanmax(smooth) - np.nanmin(smooth)),
        "curve_variance": float(np.nanvar(smooth)),
        "peak_excess_over_median": float(np.nanmax(smooth) - np.nanmedian(smooth)),
        "peak": float(window["peak"]),
    }


def _empirical_upper_p(null_values: Sequence[float], observed: float) -> float:
    arr = np.asarray(null_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not np.isfinite(observed) or len(arr) == 0:
        return np.nan
    return float((1 + np.sum(arr >= float(observed))) / (len(arr) + 1))


def _rank_quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Stable quantile bins even when many genes share the same value."""
    values = np.asarray(values, float)
    n = max(2, int(n_bins))
    ranks = pd.Series(values).rank(method="first", na_option="bottom").to_numpy(float)
    q = np.floor((ranks - 1) / max(len(ranks), 1) * n).astype(int)
    return np.clip(q, 0, n - 1)


def _sample_matched_by_2d_bins(
    rng: np.random.Generator,
    target_idx: np.ndarray,
    eligible_idx: np.ndarray,
    mean_bin: np.ndarray,
    det_bin: np.ndarray,
) -> np.ndarray:
    """Sample a no-replacement gene set matched to target counts across mean-expression/detection bins."""
    from collections import Counter

    target_idx = np.asarray(target_idx, int)
    eligible_idx = np.asarray(eligible_idx, int)
    chosen = []
    used = set()
    keys = {}
    for i in eligible_idx:
        keys.setdefault((int(mean_bin[i]), int(det_bin[i])), []).append(int(i))
    counts = Counter((int(mean_bin[i]), int(det_bin[i])) for i in target_idx)
    # Match larger strata first. Counter keeps first target occurrence for tied counts.
    for key, cnt in counts.most_common():
        cnt = int(cnt)
        pool = [i for i in keys.get(key, []) if i not in used]
        if len(pool) < cnt:
            # Expand deterministically to nearest 2D bins while preserving no replacement.
            candidates = [i for i in eligible_idx if int(i) not in used]
            candidates = sorted(
                candidates,
                key=lambda i: (
                    abs(int(mean_bin[i]) - key[0]) + abs(int(det_bin[i]) - key[1]),
                    rng.random(),
                ),
            )
            pick = candidates[:cnt]
        else:
            pick = rng.choice(np.asarray(pool, int), size=cnt, replace=False).tolist()
        chosen.extend(map(int, pick))
        used.update(map(int, pick))
    if len(chosen) != len(target_idx):
        raise RuntimeError("Matched-gene sampler failed to preserve signature size")
    return np.asarray(chosen, int)


def response_window_null_test(
    X: sparse.csr_matrix,
    genes: Sequence[str],
    meta: pd.DataFrame,
    gh: pd.DataFrame,
    n_permutations: int = 1000,
    n_bins: int = 30,
    threshold_fraction: float = 0.80,
    match_bins: int = 4,
    random_state: int = 42,
) -> Dict[str, object]:
    """Formal H1 null models for maturation-dependent structure of projected GH response potential.

    Two complementary nulls are evaluated through the complete score -> smoothing -> peak pipeline:
    (1) **weight permutation**: the selected GH genes are fixed but their GH-derived weights are permuted;
    (2) **expression/detection-matched random genes**: a same-sized background gene set is sampled with
        matching on mean expression and detection prevalence, and the observed weight distribution is
        randomly reassigned to those genes.

    The primary statistic defined in the finalized analysis is smoothed-curve amplitude, max f(tau)-min f(tau).
    Curve variance and peak excess over the curve median are reported as complementary statistics.
    Importantly, the permutation test establishes non-random developmental *structure*; it does not
    confer a p-value on the biological uniqueness of the observed peak location.
    """
    if "pseudotime" not in meta.columns:
        raise ValueError("meta must contain pseudotime")
    X = X.tocsc()
    genes = list(map(str, genes))
    lookup = gene_lookup(genes)
    g = _canonical_gene_frame(gh.copy())
    if "weight" not in g.columns:
        raise ValueError("GH table must contain normalized weight")
    g = g.loc[pd.to_numeric(g["weight"], errors="coerce").fillna(0).abs() > 0].copy()
    g["absw"] = pd.to_numeric(g["weight"], errors="coerce").abs()
    # A permutation indexes this ordered list. Winsorization creates ties, so use
    # the canonical gene symbol as the secondary key on every CPU and input order.
    g = g.sort_index().sort_values("absw", ascending=False, kind="stable")
    idx = []
    weights = []
    used = []
    for gene, row in g.iterrows():
        key = str(gene).upper()
        if key in lookup and np.isfinite(float(row["weight"])) and abs(float(row["weight"])) > 0:
            idx.append(int(lookup[key]))
            weights.append(float(row["weight"]))
            used.append(str(gene))
    idx = np.asarray(idx, int)
    weights = np.asarray(weights, float)
    means, sds, det = _sparse_gene_moments(X)
    keep = np.isfinite(sds[idx]) & (sds[idx] > EPS)
    idx = idx[keep]
    weights = weights[keep]
    used = [g for g, k in zip(used, keep) if k]
    if len(idx) < 10:
        raise ValueError("Too few selected GH genes are usable for H1 null testing")
    pt = pd.to_numeric(meta["pseudotime"], errors="coerce").to_numpy(float)
    observed_score = _sparse_standardized_weighted_score(X, idx, weights, means, sds)
    observed_window = fit_response_window(
        pt,
        observed_score,
        n_bins=n_bins,
        threshold_fraction=threshold_fraction,
        n_boot=0,
        random_state=random_state,
    )
    obs = _curve_structure_stats(observed_window)

    # Matching bins are calculated over all non-constant expressed genes.
    mean_bin = _rank_quantile_bins(means, match_bins)
    det_bin = _rank_quantile_bins(det, match_bins)
    all_idx = np.arange(X.shape[1], dtype=int)
    selected_set = set(map(int, idx))
    eligible = np.asarray(
        [
            i
            for i in all_idx
            if i not in selected_set and np.isfinite(sds[i]) and sds[i] > EPS and det[i] > 0
        ],
        int,
    )
    if len(eligible) < len(idx):
        raise ValueError("Insufficient background genes for expression/detection-matched H1 null")

    rng = np.random.default_rng(random_state)
    rows = []
    wp_curves = []
    mg_curves = []
    B = max(1, int(n_permutations))
    for b in range(B):
        # Null 1: break the mapping between GH perturbational weights and the selected genes.
        wp = weights[rng.permutation(len(weights))]
        score = _sparse_standardized_weighted_score(X, idx, wp, means, sds)
        win = fit_response_window(
            pt,
            score,
            n_bins=n_bins,
            threshold_fraction=threshold_fraction,
            n_boot=0,
            random_state=random_state + b + 1,
        )
        st = _curve_structure_stats(win)
        st.update({"null_type": "weight_permutation", "iteration": b + 1})
        rows.append(st)
        wp_curves.append(np.asarray(win["smooth"], dtype=np.float32))

        # Null 2: replace the GH genes with expression/detection-matched background genes.
        ridx = _sample_matched_by_2d_bins(rng, idx, eligible, mean_bin, det_bin)
        rw = weights[rng.permutation(len(weights))]
        score2 = _sparse_standardized_weighted_score(X, ridx, rw, means, sds)
        win2 = fit_response_window(
            pt,
            score2,
            n_bins=n_bins,
            threshold_fraction=threshold_fraction,
            n_boot=0,
            random_state=random_state + B + b + 1,
        )
        st2 = _curve_structure_stats(win2)
        st2.update({"null_type": "matched_random_genes", "iteration": b + 1})
        rows.append(st2)
        mg_curves.append(np.asarray(win2["smooth"], dtype=np.float32))

    null = pd.DataFrame(rows)
    summary = {
        "available": 1,
        "n_permutations": B,
        "n_signature_genes_used": int(len(idx)),
        "observed_amplitude": obs["amplitude"],
        "observed_curve_variance": obs["curve_variance"],
        "observed_peak_excess": obs["peak_excess_over_median"],
        "observed_peak": obs["peak"],
        "weight_permutation_p_amplitude": _empirical_upper_p(
            null.loc[null.null_type == "weight_permutation", "amplitude"], obs["amplitude"]
        ),
        "weight_permutation_p_variance": _empirical_upper_p(
            null.loc[null.null_type == "weight_permutation", "curve_variance"],
            obs["curve_variance"],
        ),
        "matched_gene_p_amplitude": _empirical_upper_p(
            null.loc[null.null_type == "matched_random_genes", "amplitude"], obs["amplitude"]
        ),
        "matched_gene_p_variance": _empirical_upper_p(
            null.loc[null.null_type == "matched_random_genes", "curve_variance"],
            obs["curve_variance"],
        ),
    }
    wp_arr = np.vstack(wp_curves)
    mg_arr = np.vstack(mg_curves)
    grid = np.asarray(observed_window["grid"], float)
    envelopes = {
        "grid": grid,
        "weight_perm_lo": np.percentile(wp_arr, 2.5, axis=0),
        "weight_perm_hi": np.percentile(wp_arr, 97.5, axis=0),
        "matched_lo": np.percentile(mg_arr, 2.5, axis=0),
        "matched_hi": np.percentile(mg_arr, 97.5, axis=0),
    }
    return {
        "summary": summary,
        "null_stats": null,
        "observed_window": observed_window,
        "observed_score": observed_score,
        "envelopes": envelopes,
        "signature_genes_used": used,
    }


def _zonal_structure_statistic(q: np.ndarray) -> float:
    q = np.asarray(q, float)
    k = len(q)
    return float(np.sqrt(np.sum((q - 1.0 / k) ** 2)))


def zonal_localization_null_test(
    spec: pd.DataFrame,
    zone_mean: pd.DataFrame,
    gh: pd.DataFrame,
    n_permutations: int = 1000,
    match_bins: int = 6,
    random_state: int = 42,
) -> Dict[str, object]:
    """Formal H2 test of non-random zonal localization using two complementary null models.

    Primary label-permutation null: independently permute the zone labels within each selected
    gene's specificity vector. This preserves each gene's degree of zonal specificity while destroying
    coherent alignment to named zones. Complementary matched-gene null: replace selected GH genes
    with a same-sized set matched on mean zonal expression and transfer the observed absolute-weight
    distribution. The global statistic is Euclidean deviation of the unsigned burden vector from a
    uniform K-zone burden.
    """
    s = _canonical_gene_frame(spec.copy())
    zm = _canonical_gene_frame(zone_mean.copy())
    g = _canonical_gene_frame(gh.copy())
    w = pd.to_numeric(
        g.get("weight", pd.Series(index=g.index, dtype=float)), errors="coerce"
    ).fillna(0.0)
    selected = w.index[w.abs() > 0]
    common = s.index.intersection(selected)
    if len(common) < 10:
        raise ValueError("Too few selected GH genes overlap the zonal dataset for H2 null testing")
    zones = list(s.columns)
    k = len(zones)
    if k < 2:
        raise ValueError("At least two zones are required for H2 null testing")
    mat = s.loc[common, zones].to_numpy(float)
    ww = w.reindex(common).abs().to_numpy(float)
    ww = ww / (ww.sum() + EPS)
    q_obs = ww @ mat
    t_obs = _zonal_structure_statistic(q_obs)

    # Background mean expression is used only for the matched-random-gene null.
    bg = s.index.intersection(zm.index)
    zm_mean = pd.to_numeric(zm.loc[bg].mean(axis=1), errors="coerce")
    finite = zm_mean[np.isfinite(zm_mean)].index
    bg = bg.intersection(finite)
    bg_vals = zm_mean.reindex(bg).to_numpy(float)
    bins = _rank_quantile_bins(bg_vals, match_bins)
    bin_map = {gene: int(b) for gene, b in zip(bg, bins)}
    selected_bins = [bin_map.get(g, None) for g in common]
    # Genes lacking a zone-mean value are assigned the median bin.
    med_bin = int(np.median(bins)) if len(bins) else 0
    selected_bins = [med_bin if b is None else int(b) for b in selected_bins]
    eligible = [g for g in bg if g not in set(common)]
    pools = {
        b: [g for g in eligible if bin_map.get(g, -1) == b] for b in range(max(2, int(match_bins)))
    }

    rng = np.random.default_rng(random_state)
    label_rows = []
    match_rows = []
    label_q = []
    match_q = []
    B = max(1, int(n_permutations))
    for b in range(B):
        # Independently shuffle the named zone identity within every GH-selected gene.
        perm = np.vstack([row[rng.permutation(k)] for row in mat])
        q = ww @ perm
        t = _zonal_structure_statistic(q)
        label_q.append(q)
        label_rows.append(
            {
                "null_type": "within_gene_zone_label_permutation",
                "iteration": b + 1,
                "structure_statistic": t,
                **{f"burden_{z}": float(q[j]) for j, z in enumerate(zones)},
            }
        )

        # Mean-expression-matched random gene set.
        chosen = []
        used = set()
        remaining = {key: list(values) for key, values in pools.items()}
        assert all(
            len(remaining.get(key, [])) >= selected_bins.count(key) for key in set(selected_bins)
        )
        for target_bin in selected_bins:
            pool = remaining.get(target_bin, [])
            if pool:
                pick = pool.pop(int(rng.integers(len(pool))))
            else:
                candidates = [g0 for g0 in eligible if g0 not in used]
                if not candidates:
                    raise RuntimeError("Insufficient matched genes for H2 null")
                pick = min(
                    candidates,
                    key=lambda g0: (abs(bin_map.get(g0, med_bin) - target_bin), rng.random()),
                )
            chosen.append(pick)
            used.add(pick)
        rmat = s.loc[chosen, zones].to_numpy(float)
        rw = ww[rng.permutation(len(ww))]
        q2 = rw @ rmat
        t2 = _zonal_structure_statistic(q2)
        match_q.append(q2)
        match_rows.append(
            {
                "null_type": "mean_expression_matched_random_genes",
                "iteration": b + 1,
                "structure_statistic": t2,
                **{f"burden_{z}": float(q2[j]) for j, z in enumerate(zones)},
            }
        )

    null = pd.DataFrame(label_rows + match_rows)
    label_arr = np.vstack(label_q)
    match_arr = np.vstack(match_q)
    summary = {
        "available": 1,
        "n_permutations": B,
        "n_signature_genes_used": int(len(common)),
        "n_zones": int(k),
        "observed_structure_statistic": t_obs,
        "label_permutation_p": _empirical_upper_p(
            null.loc[null.null_type == "within_gene_zone_label_permutation", "structure_statistic"],
            t_obs,
        ),
        "matched_gene_p": _empirical_upper_p(
            null.loc[
                null.null_type == "mean_expression_matched_random_genes", "structure_statistic"
            ],
            t_obs,
        ),
    }
    observed = pd.DataFrame(
        {
            "zone": zones,
            "observed_burden": q_obs,
            "label_null_median": np.median(label_arr, axis=0),
            "label_null_lo": np.percentile(label_arr, 2.5, axis=0),
            "label_null_hi": np.percentile(label_arr, 97.5, axis=0),
            "matched_null_median": np.median(match_arr, axis=0),
            "matched_null_lo": np.percentile(match_arr, 2.5, axis=0),
            "matched_null_hi": np.percentile(match_arr, 97.5, axis=0),
        }
    )
    return {"summary": summary, "null_stats": null, "zone_summary": observed}


def response_window_sensitivity(
    X: sparse.csr_matrix,
    genes: Sequence[str],
    meta: pd.DataFrame,
    gh: pd.DataFrame,
    signature_sizes: Sequence[int] = (100, 200, 300),
    threshold_fractions: Sequence[float] = (0.70, 0.80, 0.90),
    n_bins: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate whether the inferred GH Response Window is robust to analysis choices.

    This analysis is intentionally self-contained: it uses the already loaded single-cell matrix,
    pseudotime, and independently learned GSE3689 GH weights. It therefore cannot fail because an
    external supplementary file changes URL or becomes temporarily unavailable.

    For each requested number of top GH-signature genes N, the cell-level response-potential score
    is recomputed. For each response-window threshold q, a window is fitted using the same
    predefined definition used in the primary analysis. The returned table contains the actual
    number of GH genes found in the single-cell matrix, peak location, interval boundaries, width,
    and the Spearman correlation of each sensitivity score with the primary N=300 score when it is
    available.
    """
    if "pseudotime" not in meta.columns:
        raise ValueError("meta must contain a 'pseudotime' column")
    pt = pd.to_numeric(meta["pseudotime"], errors="coerce").to_numpy(float)
    # Primary score for concordance when available.
    try:
        primary_score, _ = weighted_gh_score(X, genes, gh, max_genes=300)
    except Exception:
        primary_score = None
    rows = []
    for requested_n in signature_sizes:
        score, used = weighted_gh_score(X, genes, gh, max_genes=int(requested_n))
        if primary_score is not None:
            ok = np.isfinite(score) & np.isfinite(primary_score)
            rho = (
                float(spearmanr(score[ok], primary_score[ok]).correlation)
                if ok.sum() > 2
                else np.nan
            )
        else:
            rho = np.nan
        for q in threshold_fractions:
            w = fit_response_window(
                pt,
                score,
                n_bins=n_bins,
                threshold_fraction=float(q),
                n_boot=0,
                random_state=random_state,
            )
            rows.append(
                {
                    "requested_signature_size": int(requested_n),
                    "genes_used_in_sc": int(len(used)),
                    "threshold_fraction": float(q),
                    "peak": float(w["peak"]),
                    "window_lower": float(w["window_lower"]),
                    "window_upper": float(w["window_upper"]),
                    "window_width": float(w["window_width"]),
                    "score_spearman_vs_primary": rho,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["threshold_fraction", "requested_signature_size"]).reset_index(
        drop=True
    )


def signature_weighting_sensitivity(
    X: sparse.csr_matrix,
    genes: Sequence[str],
    meta: pd.DataFrame,
    gh: pd.DataFrame,
    top_n: int = 300,
    methods: Sequence[str] = ("primary", "effect_only", "evidence_only"),
    threshold_fraction: float = 0.80,
    n_bins: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Test two alternative constructions of the acute-GH signature weights."""
    if "pseudotime" not in meta.columns:
        raise ValueError("meta must contain pseudotime")
    pt = pd.to_numeric(meta["pseudotime"], errors="coerce").to_numpy(float)
    scores = {}
    rows = []
    for method in methods:
        gmethod = construct_gh_weights(gh, method=method, top_n=top_n)
        score, used = weighted_gh_score(X, genes, gmethod, max_genes=top_n)
        scores[str(method)] = score
        w = fit_response_window(
            pt,
            score,
            n_bins=n_bins,
            threshold_fraction=threshold_fraction,
            n_boot=0,
            random_state=random_state,
        )
        rows.append(
            {
                "weight_method": str(method),
                "genes_used_in_sc": int(len(used)),
                "peak": float(w["peak"]),
                "window_lower": float(w["window_lower"]),
                "window_upper": float(w["window_upper"]),
                "window_width": float(w["window_width"]),
            }
        )
    primary = scores.get("primary")
    for row in rows:
        score = scores[row["weight_method"]]
        if primary is None:
            rho = np.nan
        else:
            ok = np.isfinite(score) & np.isfinite(primary)
            rho = float(spearmanr(score[ok], primary[ok]).correlation) if ok.sum() > 2 else np.nan
        row["score_spearman_vs_primary"] = rho
    return pd.DataFrame(rows)


def pseudotime_root_knn_robustness(
    pcs: np.ndarray,
    state_scores: pd.DataFrame,
    gh_response_score: Sequence[float],
    primary_pseudotime: Sequence[float],
    k_values: Sequence[int] = (10, 15, 20, 30),
    root_strategies: Sequence[str] = (
        "resting_minus_hypertrophic_max",
        "resting_score_max",
        "resting_centroid_medoid",
    ),
    threshold_fraction: float = 0.80,
    n_bins: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Quantify sensitivity of the response window to root-cell and kNN-graph choices."""
    primary = np.asarray(primary_pseudotime, float)
    score = np.asarray(gh_response_score, float)
    rows = []
    for root_strategy in root_strategies:
        for k in k_values:
            pt, graph, root = graph_geodesic_pseudotime(
                pcs, state_scores, n_neighbors=int(k), root_strategy=root_strategy
            )
            ok = np.isfinite(pt) & np.isfinite(primary)
            rho = float(spearmanr(pt[ok], primary[ok]).correlation) if ok.sum() > 2 else np.nan
            w = fit_response_window(
                pt,
                score,
                n_bins=n_bins,
                threshold_fraction=threshold_fraction,
                n_boot=0,
                random_state=random_state,
            )
            rows.append(
                {
                    "root_strategy": root_strategy,
                    "k_neighbors": int(k),
                    "root_index": int(root),
                    "finite_fraction": float(np.isfinite(pt).mean()),
                    "pseudotime_spearman_vs_primary": rho,
                    "peak": float(w["peak"]),
                    "window_lower": float(w["window_lower"]),
                    "window_upper": float(w["window_upper"]),
                    "window_width": float(w["window_width"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["root_strategy", "k_neighbors"]).reset_index(drop=True)


def _empty_conservation_table() -> pd.DataFrame:
    """Return a schema-stable empty conservation table.

    A zero-overlap mouse/human comparison is a valid data state, not a reason
    for downstream code to fail with KeyError('conservation').
    """
    df = pd.DataFrame(columns=CONSERVATION_COLUMNS)
    df.index.name = "gene"
    return df


def cross_species_conservation(
    mouse_spec: pd.DataFrame, human_spec: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compare zonal specificity between mouse and human growth plate.

    The function is deliberately *schema-stable*: even when there are no
    usable cross-species gene symbols, it returns an empty DataFrame that
    still contains the expected ``conservation`` column. This prevents the
    notebook from crashing and makes the absence of overlap explicit.
    """
    required = ["resting", "proliferative", "hypertrophic"]
    stats: Dict[str, float] = {
        "n_mouse_genes": int(len(mouse_spec)),
        "n_human_genes": int(len(human_spec)),
        "n_symbol_overlap_before_filter": 0,
        "n_ortholog_symbols": 0,
        "mean_gene_conservation": np.nan,
        "zone_profile_spearman": np.nan,
    }

    if mouse_spec is None or human_spec is None or mouse_spec.empty or human_spec.empty:
        warnings.warn(
            "Mouse-human conservation could not be computed because one zonal "
            "specificity table is empty. The analysis will continue with "
            "conservation marked unavailable."
        )
        return _empty_conservation_table(), stats

    m = mouse_spec.copy()
    h = human_spec.copy()
    m.index = pd.Index([str(x).strip().upper() for x in m.index])
    h.index = pd.Index([str(x).strip().upper() for x in h.index])

    # Upper-casing can create duplicate symbols (e.g. aliases differing only by
    # case). Collapse these deterministically before matching.
    m = m.groupby(level=0).mean(numeric_only=True)
    h = h.groupby(level=0).mean(numeric_only=True)

    missing_mouse = [z for z in required if z not in m.columns]
    missing_human = [z for z in required if z not in h.columns]
    if missing_mouse or missing_human:
        warnings.warn(
            "Mouse-human conservation skipped because required zones are missing. "
            f"mouse missing={missing_mouse}; human missing={missing_human}."
        )
        return _empty_conservation_table(), stats

    common = m.index.intersection(h.index)
    stats["n_symbol_overlap_before_filter"] = int(len(common))
    if len(common) == 0:
        warnings.warn(
            "No shared gene symbols were found between mouse and human zonal "
            "tables after normalization. Check GEO platform gene-symbol mapping. "
            "The pipeline will continue without the conservation component."
        )
        return _empty_conservation_table(), stats

    rows = []
    for g in common:
        vm = pd.to_numeric(m.loc[g, required], errors="coerce").to_numpy(dtype=float)
        vh = pd.to_numeric(h.loc[g, required], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(vm).all() and np.isfinite(vh).all()):
            continue
        # Re-normalize defensively in case upstream filtering changed row sums.
        sm, sh = float(vm.sum()), float(vh.sum())
        if sm <= 0 or sh <= 0:
            continue
        vm = vm / sm
        vh = vh / sh
        c = 1.0 - 0.5 * np.sum(np.abs(vm - vh))
        rows.append(
            {
                "gene": g,
                "conservation": float(np.clip(c, 0, 1)),
                "mouse_resting": vm[0],
                "mouse_proliferative": vm[1],
                "mouse_hypertrophic": vm[2],
                "human_resting": vh[0],
                "human_proliferative": vh[1],
                "human_hypertrophic": vh[2],
            }
        )

    if not rows:
        warnings.warn(
            "Shared symbols were present, but none had complete finite zonal "
            "profiles. The pipeline will continue without the conservation component."
        )
        return _empty_conservation_table(), stats

    df = pd.DataFrame(rows).set_index("gene")
    # Keep a deterministic column order.
    df = df.reindex(columns=CONSERVATION_COLUMNS)
    stats["n_ortholog_symbols"] = int(len(df))
    stats["mean_gene_conservation"] = float(df["conservation"].mean())

    # Global zone-profile correspondence across all matched genes. A Spearman
    # coefficient based on only three aggregate zone values is descriptive.
    mv = np.array(
        [
            df["mouse_resting"].mean(),
            df["mouse_proliferative"].mean(),
            df["mouse_hypertrophic"].mean(),
        ]
    )
    hv = np.array(
        [
            df["human_resting"].mean(),
            df["human_proliferative"].mean(),
            df["human_hypertrophic"].mean(),
        ]
    )
    stats["zone_profile_spearman"] = float(spearmanr(mv, hv).statistic)
    return df, stats


def window_gene_statistics(
    X: sparse.csr_matrix, genes: Sequence[str], meta: pd.DataFrame, lower: float, upper: float
) -> pd.DataFrame:
    """Gene-level contrast between cells inside and outside the GH Response Window.

    The original implementation used a raw difference of mean log-expression.
    That can favor abundant housekeeping genes.  The revised candidate-ranking
    method therefore uses a standardized mean difference (SMD) together with
    expression prevalence, while retaining raw means for auditability.
    """
    pt = pd.to_numeric(meta["pseudotime"], errors="coerce").to_numpy(float)
    inside = np.isfinite(pt) & (pt >= float(lower)) & (pt <= float(upper))
    outside = np.isfinite(pt) & ~inside
    if inside.sum() < 10 or outside.sum() < 10:
        return pd.DataFrame(
            columns=[
                "mean_inside",
                "mean_outside",
                "window_minus_outside_logexpr",
                "window_smd",
                "detected_fraction_inside",
                "detected_fraction_outside",
                "detected_fraction_max",
                "n_inside",
                "n_outside",
            ]
        )
    Xi = X[inside].tocsr()
    Xo = X[outside].tocsr()
    ni, no = Xi.shape[0], Xo.shape[0]
    mi = np.asarray(Xi.mean(axis=0)).ravel().astype(float)
    mo = np.asarray(Xo.mean(axis=0)).ravel().astype(float)
    m2i = np.asarray(Xi.power(2).mean(axis=0)).ravel().astype(float)
    m2o = np.asarray(Xo.power(2).mean(axis=0)).ravel().astype(float)
    vi = np.maximum(m2i - mi**2, 0.0)
    vo = np.maximum(m2o - mo**2, 0.0)
    pooled = np.sqrt(
        np.maximum(((max(ni - 1, 1) * vi) + (max(no - 1, 1) * vo)) / max(ni + no - 2, 1), 0.0)
    )
    smd = (mi - mo) / (pooled + EPS)
    di = np.asarray((Xi > 0).mean(axis=0)).ravel().astype(float)
    do = np.asarray((Xo > 0).mean(axis=0)).ravel().astype(float)
    idx = pd.Index([str(g).upper() for g in genes], name="gene")
    out = pd.DataFrame(
        {
            "mean_inside": mi,
            "mean_outside": mo,
            "window_minus_outside_logexpr": mi - mo,
            "window_smd": smd,
            "detected_fraction_inside": di,
            "detected_fraction_outside": do,
            "detected_fraction_max": np.maximum(di, do),
            "n_inside": int(ni),
            "n_outside": int(no),
        },
        index=idx,
    )
    if out.index.has_duplicates:
        numeric = out.groupby(level=0).mean(numeric_only=True)
        out = numeric
    return out


def priority_table(
    gh: pd.DataFrame,
    window_evidence,
    conservation: pd.DataFrame,
    height_candidates: Sequence[str],
    min_prevalence: float = 0.05,
) -> pd.DataFrame:
    """Rank GH-growth candidates with a GSE3689-anchored candidate universe.

    Critical consistency rules
    --------------------------
    1. Only genes actually selected into the primary/alternative GH signature
       (``selected_signature`` or non-zero ``weight``) may enter the ranking.
    2. Acute-GH evidence is the absolute *same perturbational weight* used for
       single-cell projection (``weight_raw``), not a separate |Z| statistic.
    3. Window evidence is a standardized mean difference, not raw expression
       difference, and candidates must be detectably expressed in the atlas.
    4. GH and window evidence are mandatory and define the primary MPS for
       every eligible gene. Cross-species similarity and height genetics remain
       separate supporting/validation streams and cannot change the primary rank.
    """
    g = _canonical_gene_frame(gh.copy())
    if "selected_signature" in g.columns:
        selected = g["selected_signature"].fillna(False).astype(bool)
    elif "weight" in g.columns:
        selected = pd.to_numeric(g["weight"], errors="coerce").fillna(0).abs() > 0
    else:
        raise ValueError("GH table must contain selected_signature or weight")
    g = g.loc[selected].copy()
    if g.empty:
        return pd.DataFrame()

    if isinstance(window_evidence, pd.DataFrame):
        ws = _canonical_gene_frame(window_evidence.copy())
    elif isinstance(window_evidence, pd.Series):
        ser = _canonical_gene_series(window_evidence.copy())
        ws = pd.DataFrame({"window_smd": ser, "detected_fraction_max": 1.0})
    else:
        ws = pd.DataFrame()
    if "window_smd" not in ws.columns:
        return pd.DataFrame()
    if "detected_fraction_max" not in ws.columns:
        ws["detected_fraction_max"] = 1.0

    # Candidate universe is anchored in the selected GH signature and requires
    # measurable single-cell window evidence.
    idx = g.index.intersection(ws.index)
    out = pd.DataFrame(index=idx)
    out["selected_primary_signature"] = True
    out["gh_weight_raw_abs"] = pd.to_numeric(g["weight_raw"], errors="coerce").abs().reindex(idx)
    out["gh_weight_normalized_abs"] = pd.to_numeric(g["weight"], errors="coerce").abs().reindex(idx)
    out["gh_abs_z"] = (
        pd.to_numeric(g["z_combined"], errors="coerce").abs().reindex(idx)
        if "z_combined" in g.columns
        else np.nan
    )
    out["gh_mean_log2fc"] = (
        pd.to_numeric(g["mean_log2fc"], errors="coerce").reindex(idx)
        if "mean_log2fc" in g.columns
        else np.nan
    )
    out["window_smd"] = pd.to_numeric(ws["window_smd"], errors="coerce").reindex(idx)
    out["window_effect_abs"] = out["window_smd"].abs()
    out["window_detected_fraction_max"] = pd.to_numeric(
        ws["detected_fraction_max"], errors="coerce"
    ).reindex(idx)
    out = out.loc[
        (out["window_detected_fraction_max"] >= float(min_prevalence))
        & out["gh_weight_raw_abs"].notna()
        & out["window_effect_abs"].notna()
    ].copy()
    if out.empty:
        return out

    cons = (
        _canonical_gene_frame(conservation.copy())
        if isinstance(conservation, pd.DataFrame) and not conservation.empty
        else pd.DataFrame()
    )
    has_conservation = not cons.empty and "conservation" in cons.columns
    out["conservation"] = (
        pd.to_numeric(cons["conservation"], errors="coerce").reindex(out.index)
        if has_conservation
        else np.nan
    )
    hset = set(map(str.upper, height_candidates or []))
    out["height_supported"] = out.index.to_series().isin(hset) if hset else False

    # Percentiles are calculated *within the actual GH-selected candidate
    # universe*, so a non-selected GSE3689 gene can no longer outrank a selected
    # signature gene simply because it has a large z statistic.
    out["P_GH"] = percentile_score(out["gh_weight_raw_abs"])
    out["P_window"] = percentile_score(out["window_effect_abs"])
    out["P_conservation"] = percentile_score(out["conservation"])

    def geom_primary(row):
        if pd.isna(row["P_GH"]) or pd.isna(row["P_window"]):
            return np.nan
        vals = [float(row["P_GH"]), float(row["P_window"])]
        vals = [max(v, EPS) for v in vals]
        return float(np.exp(np.mean(np.log(vals))))

    def geom_cross_species(row):
        if pd.isna(row["P_GH"]) or pd.isna(row["P_window"]) or pd.isna(row["P_conservation"]):
            return np.nan
        vals = [max(float(row[k]), EPS) for k in ["P_GH", "P_window", "P_conservation"]]
        return float(np.exp(np.mean(np.log(vals))))

    # The primary score always contains the same two components. This avoids
    # comparing two-component and three-component geometric means in one rank.
    out["MPS"] = out.apply(geom_primary, axis=1)
    out["MPS_cross_species"] = out.apply(geom_cross_species, axis=1)
    out["n_primary_components"] = 2
    out["rank_MPS"] = out["MPS"].rank(ascending=False, method="min")
    out["rank_MPS_cross_species"] = out["MPS_cross_species"].rank(ascending=False, method="min")
    out["rank_GH"] = out["P_GH"].rank(ascending=False, method="min")
    out["rank_window"] = out["P_window"].rank(ascending=False, method="min")
    out["rank_conservation"] = out["P_conservation"].rank(ascending=False, method="min")
    out["canonical_GH_IGF"] = out.index.to_series().isin(CANONICAL_GH_IGF)
    out["latent_candidate"] = ~out["canonical_GH_IGF"]
    out["genetics_validated_latent"] = out["latent_candidate"] & out["height_supported"]
    return out.sort_values(["MPS", "gh_weight_raw_abs"], ascending=[False, False])


def candidate_ranking_sensitivity(
    X: sparse.csr_matrix,
    genes: Sequence[str],
    meta: pd.DataFrame,
    gh_stats: pd.DataFrame,
    conservation: pd.DataFrame,
    methods: Sequence[str] = ("primary", "effect_only", "evidence_only"),
    top_n: int = 300,
    top_k: int = 20,
    threshold_fraction: float = 0.80,
    min_prevalence: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute the *entire candidate-ranking chain* for each GH weighting rule.

    Each method gets its own selected signature, projected cell score, Response
    Window, standardized window-gene statistics, and MPS.  This avoids the
    misleading practice of changing GH weights while keeping a candidate table
    derived from the primary preprocessing.
    """
    if "pseudotime" not in meta.columns:
        raise ValueError("meta must contain pseudotime")
    pt = pd.to_numeric(meta["pseudotime"], errors="coerce").to_numpy(float)
    tables = {}
    long_rows = []
    for method in methods:
        gm = construct_gh_weights(gh_stats, method=str(method), top_n=int(top_n))
        score, _ = weighted_gh_score(X, genes, gm, max_genes=int(top_n))
        win = fit_response_window(
            pt,
            score,
            threshold_fraction=float(threshold_fraction),
            n_boot=0,
            random_state=random_state,
        )
        meta_m = meta.copy()
        meta_m["gh_response_score"] = score
        ws = window_gene_statistics(X, genes, meta_m, win["window_lower"], win["window_upper"])
        pr = priority_table(gm, ws, conservation, [], min_prevalence=min_prevalence)
        pr = pr.loc[pr["MPS"].notna()].copy() if not pr.empty else pr
        if not pr.empty:
            pr["candidate_rank"] = np.arange(1, len(pr) + 1)
            pr["weight_method"] = str(method)
            for gene, row in pr.iterrows():
                long_rows.append(
                    {
                        "gene": gene,
                        "weight_method": str(method),
                        "candidate_rank": int(row["candidate_rank"]),
                        "MPS": float(row["MPS"]),
                        "peak": float(win["peak"]),
                        "window_lower": float(win["window_lower"]),
                        "window_upper": float(win["window_upper"]),
                    }
                )
        tables[str(method)] = pr

    long = pd.DataFrame(long_rows)
    primary = tables.get("primary", pd.DataFrame())
    summary_rows = []
    primary_top = set(primary.head(int(top_k)).index) if not primary.empty else set()
    for method in methods:
        tbl = tables.get(str(method), pd.DataFrame())
        if method == "primary":
            rho = 1.0 if not tbl.empty else np.nan
            jac = 1.0 if primary_top else np.nan
            common_n = len(tbl)
        elif primary.empty or tbl.empty:
            rho = jac = np.nan
            common_n = 0
        else:
            common = primary.index.intersection(tbl.index)
            common_n = len(common)
            rho = (
                float(
                    spearmanr(
                        primary.loc[common, "rank_MPS"], tbl.loc[common, "rank_MPS"]
                    ).correlation
                )
                if common_n >= 3
                else np.nan
            )
            tset = set(tbl.head(int(top_k)).index)
            jac = (
                float(len(primary_top & tset) / len(primary_top | tset))
                if (primary_top | tset)
                else np.nan
            )
        summary_rows.append(
            {
                "weight_method": str(method),
                "n_ranked": int(len(tbl)),
                "n_common_with_primary": int(common_n),
                "rank_spearman_vs_primary": rho,
                "top_k": int(top_k),
                "top_k_jaccard_vs_primary": jac,
            }
        )
    summary = pd.DataFrame(summary_rows)

    # Figure matrix is indexed by the primary top-k candidates so readers can
    # see exactly how each candidate moves under alternative GH construction.
    matrix = pd.DataFrame(index=list(primary.head(int(top_k)).index))
    for method in methods:
        tbl = tables.get(str(method), pd.DataFrame())
        matrix[str(method)] = tbl["rank_MPS"].reindex(matrix.index) if not tbl.empty else np.nan
    matrix.index.name = "gene"
    return long, summary, matrix


def test_gse3689_shift_and_lowess():
    rng = np.random.default_rng(7)
    n = 2500
    ids = np.arange(1, n + 1).astype(str)
    A = rng.uniform(6.5, 10.0, n)
    true_m = rng.normal(0, 0.12, n)
    true_m[:30] += 1.0
    bias = 0.22 * (A - A.mean())
    obs_m = true_m + bias
    fg1 = 2 ** (A + obs_m / 2)
    fg2 = 2 ** (A - obs_m / 2)
    b1 = np.full(n, 35.0)
    b2 = np.full(n, 42.0)
    ch1med = fg1 + b1
    ch2med = fg2 + b2
    # Mimic the historical family-SOFT one-column shift.
    df = pd.DataFrame(
        {
            "ID_REF": ids,
            "VALUE": ch1med + 0.3,
            "CH1MeanSig": ch2med + 0.3,
            "CH2MeanSig": ch1med,
            "CH1MedianSig": ch2med,
            "CH2MedianSig": b1 + 0.1,
            "CH1MeanBac": b2 + 0.1,
            "CH2MeanBac": b1,
            "CH1MedianBac": b2,
            "CH2MedianBac": np.nan,
            "PRE_VALUE": np.nan,
        }
    )
    meta = {
        "Sample_treatment_protocol_ch1": [
            "GH (4mg/kg body weight) injected subcutaneously; 6 hours"
        ],
        "Sample_treatment_protocol_ch2": ["PBS injected subcutaneously"],
        "Sample_label_ch1": ["Cy5"],
        "Sample_label_ch2": ["Cy3"],
    }
    est, qc = _gse3689_lowess_logratio(df, meta, raw_signal_threshold=100, lowess_frac=0.30)
    ok = np.isfinite(est.to_numpy())
    rho = np.corrcoef(est.to_numpy()[ok], true_m[ok])[0, 1]
    assert qc["soft_column_shift_repaired"] is True
    assert qc["channel1_treatment"] == "GH" and qc["channel2_treatment"] == "PBS"
    assert abs(qc["median_M_after_lowess"]) < 0.08
    assert rho > 0.80, rho


def test_gse3689_official_row_alignment():
    # Regression row transcribed from the public GEO GSM85325 table. GEO shows
    # an 11-column header but the row contains no processed VALUE/PRE_VALUE.
    # After pandas-style left alignment, the repair must recover the documented
    # raw channel fields: CH1 median=129.0, CH2 median=78.5, CH1 median
    # background=40.0, CH2 median background=60.0.
    df = pd.DataFrame(
        {
            "ID_REF": ["2"],
            "VALUE": [129.52],
            "CH1MeanSig": [78.77],
            "CH2MeanSig": [129.0],
            "CH1MedianSig": [78.5],
            "CH2MedianSig": [40.31],
            "CH1MeanBac": [60.25],
            "CH2MeanBac": [40.0],
            "CH1MedianBac": [60.0],
            "CH2MedianBac": [np.nan],
            "PRE_VALUE": [np.nan],
        }
    )
    fixed, shifted = _repair_gse3689_channel_columns(df)
    assert shifted is True
    assert float(fixed.loc[0, "CH1MedianSig"]) == 129.0
    assert float(fixed.loc[0, "CH2MedianSig"]) == 78.5
    assert float(fixed.loc[0, "CH1MedianBac"]) == 40.0
    assert float(fixed.loc[0, "CH2MedianBac"]) == 60.0


def test_candidate_universe_is_gh_selected():
    genes = ["A", "B", "C", "D", "E"]
    gh = pd.DataFrame(index=genes)
    gh["z_combined"] = [2, 3, 4, 100, 90]
    gh["mean_log2fc"] = [0.5, 0.4, 0.3, 0.1, 0.1]
    gh["weight_raw"] = [5, 4, 3, 100, 90]
    gh["weight"] = [0.7, 0.6, 0.4, 0, 0]
    gh["selected_signature"] = [True, True, True, False, False]
    ws = pd.DataFrame(index=genes)
    ws["window_smd"] = [1, 0.8, 0.6, 10, 9]
    ws["detected_fraction_max"] = 1.0
    cons = pd.DataFrame({"conservation": [0.8, 0.7, 0.9, 0.99, 0.99]}, index=genes)
    pr = priority_table(gh, ws, cons, [])
    assert set(pr.index) == {"A", "B", "C"}, pr.index.tolist()
    assert "D" not in pr.index and "E" not in pr.index
    assert pr["selected_primary_signature"].all()
    assert pr["n_primary_components"].eq(2).all()
    expected = np.sqrt(pr["P_GH"] * pr["P_window"])
    assert np.allclose(pr["MPS"], expected)
    assert pr["MPS_cross_species"].notna().all()


def test_candidate_ranking_sensitivity_runs():
    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(80)]
    gh = pd.DataFrame(index=genes)
    gh["mean_log2fc"] = rng.normal(0, 0.4, len(genes))
    gh["z_combined"] = gh["mean_log2fc"] / 0.15 + rng.normal(0, 0.2, len(genes))
    gh["analysis_eligible"] = True
    gh = construct_gh_weights(gh, "primary", top_n=30)
    X = sparse.csr_matrix(np.maximum(0, rng.normal(1, 0.7, (240, len(genes)))).astype(np.float32))
    pt = np.linspace(0, 1, 240)
    meta = pd.DataFrame({"pseudotime": pt})
    cons = pd.DataFrame({"conservation": rng.uniform(0.4, 1, len(genes))}, index=genes)
    long, summary, matrix = candidate_ranking_sensitivity(
        X, genes, meta, gh, cons, top_n=30, top_k=10
    )
    assert len(summary) == 3
    assert not matrix.empty
    assert set(summary["weight_method"]) == {"primary", "effect_only", "evidence_only"}


def test_h1_null_model_runs():
    rng = np.random.default_rng(11)
    n_cells = 220
    n_genes = 120
    genes = [f"G{i}" for i in range(n_genes)]
    pt = np.linspace(0, 1, n_cells)
    X = np.maximum(0, rng.normal(1, 0.6, (n_cells, n_genes))).astype(np.float32)
    # Inject a maturation-dependent pattern into the selected GH genes.
    gh = pd.DataFrame(index=genes)
    gh["mean_log2fc"] = rng.normal(0, 0.3, n_genes)
    gh["z_combined"] = gh["mean_log2fc"] / 0.12
    gh["analysis_eligible"] = True
    gh = construct_gh_weights(gh, "primary", top_n=30)
    tied = gh.index[gh.selected_signature][:12]
    gh.loc[tied, "weight"] = np.where(np.arange(len(tied)) % 2 == 0, 1.0, -1.0)
    selected = np.where(gh.reindex(genes)["selected_signature"].fillna(False).to_numpy())[0]
    X[:, selected[:10]] += (2.0 * np.exp(-(((pt - 0.7) / 0.12) ** 2)))[:, None]
    X = sparse.csr_matrix(X)
    meta = pd.DataFrame({"pseudotime": pt})
    res = response_window_null_test(X, genes, meta, gh, n_permutations=30, random_state=11)
    assert len(res["null_stats"]) == 60
    assert 0 <= res["summary"]["weight_permutation_p_amplitude"] <= 1
    assert 0 <= res["summary"]["matched_gene_p_amplitude"] <= 1
    expected = sorted(
        gh.index[gh.selected_signature], key=lambda gene: (-abs(gh.at[gene, "weight"]), gene)
    )
    assert res["signature_genes_used"] == expected
    reversed_rows = response_window_null_test(
        X, genes, meta, gh.iloc[::-1], n_permutations=30, random_state=11
    )
    assert reversed_rows["signature_genes_used"] == expected
    pd.testing.assert_frame_equal(res["null_stats"], reversed_rows["null_stats"], check_exact=True)
    centers = np.linspace(0, 1, 12)
    bin_scores = np.array([0, 0, 0, 0.3, 0.7, 0.7, 0.2, -0.5, 0.1, 0, 0, 0])
    grid = np.linspace(0, 1, 100)
    np.testing.assert_array_equal(
        smooth_binned_profile(centers, bin_scores, grid),
        smooth_binned_profile(centers, bin_scores + 1e-13, grid),
    )


def test_h2_null_model_runs():
    rng = np.random.default_rng(12)
    genes = [f"G{i}" for i in range(100)]
    zone_mean = pd.DataFrame(
        rng.lognormal(2, 0.4, (100, 3)),
        index=genes,
        columns=["resting", "proliferative", "hypertrophic"],
    )
    spec = zone_specificity(zone_mean)
    gh = pd.DataFrame(index=genes)
    gh["mean_log2fc"] = rng.normal(0, 0.3, 100)
    gh["z_combined"] = gh["mean_log2fc"] / 0.12
    gh["analysis_eligible"] = True
    gh = construct_gh_weights(gh, "primary", top_n=30)
    res = zonal_localization_null_test(spec, zone_mean, gh, n_permutations=30, random_state=12)
    assert len(res["null_stats"]) == 60
    assert 0 <= res["summary"]["label_permutation_p"] <= 1
    assert 0 <= res["summary"]["matched_gene_p"] <= 1


def run_regression_checks():
    test_gse3689_shift_and_lowess()
    test_gse3689_official_row_alignment()
    test_candidate_universe_is_gh_selected()
    test_candidate_ranking_sensitivity_runs()
    test_h1_null_model_runs()
    test_h2_null_model_runs()
    print("Six implementation checks passed.")


def fig_gse3689_preprocessing_qc(qc: pd.DataFrame, stem: str = "figS02_gse3689_preprocessing_qc"):
    """Supplementary audit of published-target direction/fold-change recovery."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    d = (
        qc.dropna(subset=["published_log2fc", "estimated_log2fc"]).copy()
        if qc is not None
        else pd.DataFrame()
    )
    if d.empty:
        ax.text(0.5, 0.5, "Published-target QC unavailable", ha="center", va="center")
        ax.axis("off")
    else:
        ax.scatter(d["published_log2fc"], d["estimated_log2fc"], s=42)
        lim = max(
            0.5,
            float(
                np.nanmax(np.abs(np.r_[d["published_log2fc"].values, d["estimated_log2fc"].values]))
            )
            * 1.15,
        )
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1)
        for _, r in d.iterrows():
            ax.annotate(
                f"{r['gene']} {r['time']}",
                (r["published_log2fc"], r["estimated_log2fc"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
        rho = (
            float(spearmanr(d["published_log2fc"], d["estimated_log2fc"]).correlation)
            if len(d) >= 3
            else np.nan
        )
        ax.set_xlabel("Published log2 fold-change")
        ax.set_ylabel("Reanalysis log2 fold-change")
        ax.set_title(
            f"GSE3689 preprocessing audit against published targets\nSpearman rho={rho:.2f}"
            if np.isfinite(rho)
            else "GSE3689 preprocessing audit against published targets"
        )
    export_panel(fig, stem)
    plt.close(fig)


def fig_response_window_null_test(
    result: Mapping[str, object],
):
    """Figure 14: observed H1 developmental structure versus two formal null distributions."""
    null = result["null_stats"]
    win = result["observed_window"]
    env = result["envelopes"]
    summ = result["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    ax = axes[0]
    grid = np.asarray(win["grid"], float)
    smooth = np.asarray(win["smooth"], float)
    ax.plot(grid, env["weight_perm_lo"], lw=1, ls=":", label="Weight-permutation 95% null envelope")
    ax.plot(grid, env["weight_perm_hi"], lw=1, ls=":")
    ax.plot(grid, smooth, lw=2, label="Observed GH projection")
    ax.axvline(float(win["peak"]), ls="--", lw=1)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Smoothed signed projection")
    ax.set_title("Observed developmental profile")
    ax.legend(fontsize=7)
    for ax, nt, title, pkey in [
        (
            axes[1],
            "weight_permutation",
            "Weight-permutation null",
            "weight_permutation_p_amplitude",
        ),
        (
            axes[2],
            "matched_random_genes",
            "Expression/detection-matched genes",
            "matched_gene_p_amplitude",
        ),
    ]:
        vals = (
            pd.to_numeric(null.loc[null.null_type == nt, "amplitude"], errors="coerce")
            .dropna()
            .to_numpy(float)
        )
        ax.hist(vals, bins=30)
        ax.axvline(float(summ["observed_amplitude"]), lw=2, ls="--")
        ax.set_xlabel(r"$T_{H1}=\max f-\min f$")
        ax.set_ylabel("Null realizations")
        ax.set_title(f"{title}\nempirical p={float(summ[pkey]):.4f}")
    fig.suptitle(
        "H1 formal null test: maturation-dependent structure of the GH-signature projection", y=1.03
    )
    fig.tight_layout()
    export_panel(fig, "fig14_h1_developmental_structure_null")
    plt.close(fig)


def fig_zonal_localization_null_test(
    result: Mapping[str, object], stem: str = "fig15_h2_zonal_localization_null"
):
    """Figure 15: H2 observed zonal burden versus label-permutation and matched-gene nulls."""
    summ = result["summary"]
    zs = result["zone_summary"]
    null = result["null_stats"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    x = np.arange(len(zs))
    obs = zs["observed_burden"].to_numpy(float)
    axes[0].bar(x, obs, width=0.55, label="Observed")
    axes[0].errorbar(
        x - 0.14,
        zs["label_null_median"],
        yerr=[
            zs["label_null_median"] - zs["label_null_lo"],
            zs["label_null_hi"] - zs["label_null_median"],
        ],
        fmt="o",
        capsize=3,
        label="Label-permutation null",
    )
    axes[0].errorbar(
        x + 0.14,
        zs["matched_null_median"],
        yerr=[
            zs["matched_null_median"] - zs["matched_null_lo"],
            zs["matched_null_hi"] - zs["matched_null_median"],
        ],
        fmt="s",
        capsize=3,
        label="Matched-gene null",
    )
    axes[0].set_xticks(x, zs["zone"], rotation=20)
    axes[0].set_ylabel("Unsigned GH-signature burden")
    axes[0].set_title("Observed burden and 95% null intervals")
    axes[0].legend(fontsize=7)
    for ax, nt, title, pkey in [
        (
            axes[1],
            "within_gene_zone_label_permutation",
            "Within-gene zone-label permutation",
            "label_permutation_p",
        ),
        (
            axes[2],
            "mean_expression_matched_random_genes",
            "Mean-expression-matched random genes",
            "matched_gene_p",
        ),
    ]:
        vals = (
            pd.to_numeric(null.loc[null.null_type == nt, "structure_statistic"], errors="coerce")
            .dropna()
            .to_numpy(float)
        )
        ax.hist(vals, bins=30)
        ax.axvline(float(summ["observed_structure_statistic"]), lw=2, ls="--")
        ax.set_xlabel(r"$T_{H2}=\|Q-1/K\|_2$")
        ax.set_ylabel("Null realizations")
        ax.set_title(f"{title}\nempirical p={float(summ[pkey]):.4f}")
    fig.suptitle("H2 formal null test: non-random growth-plate zonal localization", y=1.03)
    fig.tight_layout()
    export_panel(fig, stem)
    plt.close(fig)


def fig_response_window_sensitivity(
    sens: pd.DataFrame,
):
    """Publication-grade robustness figure used as primary Figure 8.

    Each point is the estimated peak pseudotime and each vertical segment is the corresponding
    high-response interval. Marker shape identifies the threshold definition. This replaces the
    former externally dependent height-candidate Figure 8.
    """
    if sens is None or sens.empty:
        raise ValueError("Response-window sensitivity table is empty")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    thresholds = sorted(pd.unique(sens["threshold_fraction"]))
    markers = ["o", "s", "^"]
    for j, q in enumerate(thresholds):
        d = sens.loc[np.isclose(sens["threshold_fraction"].astype(float), float(q))].sort_values(
            "requested_signature_size"
        )
        x = d["requested_signature_size"].to_numpy(float)
        y = d["peak"].to_numpy(float)
        lower = y - d["window_lower"].to_numpy(float)
        upper = d["window_upper"].to_numpy(float) - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            marker=markers[j % len(markers)],
            capsize=4,
            linewidth=1.6,
            label=f"{int(round(100 * q))}% interval threshold",
        )
    ax.axhline(0.5, linestyle=":", linewidth=1)
    ax.set_xlabel("Requested number of ranked whole-bone genes")
    ax.set_ylabel("Geodesic pseudotime")
    ax.set_ylim(0, 1)
    ax.set_title("Fitted maximum across signature sizes and thresholds")
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.02,
        "Point: fitted maximum. Bar: connected high-score interval.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    export_panel(fig, "fig08_response_window_sensitivity")
    plt.close(fig)


def fig_signature_weighting_sensitivity(
    sens: pd.DataFrame,
):
    if sens is None or sens.empty:
        raise ValueError("Signature-weighting sensitivity table is empty")
    labels = {
        "primary": "Primary: Z x effect",
        "effect_only": "Effect only",
        "evidence_only": "Signed Z only",
    }
    d = sens.copy()
    x = np.arange(len(d))
    y = d["peak"].to_numpy(float)
    lower = y - d["window_lower"].to_numpy(float)
    upper = d["window_upper"].to_numpy(float) - y
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", capsize=5, linewidth=1.6)
    ax.set_xticks(x, [labels.get(v, v) for v in d["weight_method"]], rotation=12, ha="right")
    ax.set_ylabel("Geodesic pseudotime")
    ax.set_ylim(0, 1)
    ax.set_title("Fitted maximum across whole-bone weight definitions")
    for i, row in d.reset_index(drop=True).iterrows():
        rho = row.get("score_spearman_vs_primary", np.nan)
        if np.isfinite(rho):
            ax.text(
                i, min(0.98, row["window_upper"] + 0.04), f"rho={rho:.2f}", ha="center", fontsize=9
            )
    export_panel(fig, "fig11_signature_weighting_sensitivity")
    plt.close(fig)


def fig_pseudotime_root_knn_robustness(
    sens: pd.DataFrame,
):
    if sens is None or sens.empty:
        raise ValueError("Pseudotime robustness table is empty")
    labels = {
        "resting_minus_hypertrophic_max": "Resting - hypertrophic max",
        "resting_score_max": "Resting-score max",
        "resting_centroid_medoid": "Resting top-decile medoid",
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    markers = ["o", "s", "^"]
    for j, strategy in enumerate(pd.unique(sens["root_strategy"])):
        d = sens[sens["root_strategy"] == strategy].sort_values("k_neighbors")
        x = d["k_neighbors"].to_numpy(float)
        y = d["peak"].to_numpy(float)
        lower = y - d["window_lower"].to_numpy(float)
        upper = d["window_upper"].to_numpy(float) - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            marker=markers[j % len(markers)],
            capsize=4,
            linewidth=1.4,
            label=labels.get(strategy, strategy),
        )
    ax.set_xlabel("k in k-nearest-neighbor graph")
    ax.set_ylabel("Pseudotime of the fitted maximum")
    ax.set_ylim(0, 1)
    ax.set_title("Pseudotime robustness to root definition and neighborhood size")
    ax.legend(frameon=False, fontsize=9)
    export_panel(fig, "fig12_pseudotime_root_knn_robustness")
    plt.close(fig)


def fig_candidate_rank_robustness(
    rank_matrix: pd.DataFrame, stem: str = "fig13_candidate_rank_robustness"
):
    """Show how primary top candidates move under alternative GH weight constructions."""
    fig, ax = plt.subplots(figsize=(7.2, max(5.5, 0.30 * max(len(rank_matrix), 1) + 2.0)))
    if rank_matrix is None or rank_matrix.empty:
        ax.text(0.5, 0.5, "Candidate-rank robustness unavailable", ha="center", va="center")
        ax.axis("off")
    else:
        d = rank_matrix.copy().astype(float)
        display = d.copy()
        finite = display.to_numpy(float)[np.isfinite(display.to_numpy(float))]
        missing_rank = float(np.nanmax(finite) + 10) if finite.size else 100.0
        arr = display.fillna(missing_rank).to_numpy(float)
        im = ax.imshow(arr, aspect="auto")
        fig.colorbar(im, ax=ax, label="Candidate rank (lower is better)")
        labels = {
            "primary": "Primary",
            "effect_only": "Effect only",
            "evidence_only": "Signed Z only",
        }
        ax.set_xticks(
            range(len(display.columns)),
            [labels.get(c, c) for c in display.columns],
            rotation=15,
            ha="right",
        )
        ax.set_yticks(range(len(display.index)), display.index)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = display.iloc[i, j]
                ax.text(
                    j,
                    i,
                    "NA" if not np.isfinite(val) else str(int(round(val))),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_title("Candidate-ranking robustness to GH-signature construction")
    export_panel(fig, stem)
    plt.close(fig)
