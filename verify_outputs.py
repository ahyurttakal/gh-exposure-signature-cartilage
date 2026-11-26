"""Compare a full reproduction with the archived release or a second run."""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse


ATOL = 1e-8
RTOL = 1e-8


def table_bytes(path):
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def portable_csv_equal(reference, candidate):
    """Match labels exactly and allow floating-point roundoff in numeric fields."""
    left, right = (pd.read_csv(p, dtype=str, keep_default_na=False) for p in (reference, candidate))
    if left.shape != right.shape or not left.columns.equals(right.columns):
        return False
    for column in left.columns:
        different = left[column] != right[column]
        if not different.any():
            continue
        x, y = left.loc[different, column], right.loc[different, column]
        # Preserve integer counts and identifiers exactly, including leading zeros.
        if (x.str.fullmatch(r"[+-]?[0-9]+") & y.str.fullmatch(r"[+-]?[0-9]+")).any():
            return False
        u, v = (pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) for values in (x, y))
        if not (
            np.isfinite(u).all()
            and np.isfinite(v).all()
            and np.allclose(u, v, rtol=RTOL, atol=ATOL)
        ):
            return False
    return True


def same_matrix(reference, candidate, portable):
    x, y = (sparse.load_npz(p).tocsr() for p in (reference, candidate))
    if x.shape != y.shape:
        return False
    if not portable:
        return (x != y).nnz == 0
    for matrix in (x, y):
        matrix.sort_indices()
    if not (np.array_equal(x.indptr, y.indptr) and np.array_equal(x.indices, y.indices)):
        return False
    for start in range(0, x.nnz, 500_000):
        block = slice(start, start + 500_000)
        if not np.allclose(x.data[block], y.data[block], rtol=RTOL, atol=ATOL):
            return False
    return True


REFERENCE_TABLES = (
    "table1.csv",
    "table2.csv",
    "H1_figure_values.csv",
    "reproduction_audit.csv",
    "revision_audit.csv",
)
HEADERS = {".pdf": b"%PDF-", ".eps": b"%!PS", ".png": b"\x89PNG\r\n\x1a\n"}


def check_inventory(run, reference):
    """Use the inventory frozen with the manuscript, independently of the new run."""
    missing = []
    items = (reference / "expected_files.txt").read_text(encoding="utf-8").splitlines()
    for name in items:
        path = run / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
        elif path.suffix in HEADERS:
            expected = HEADERS[path.suffix]
            with path.open("rb") as stream:
                if stream.read(len(expected)) != expected:
                    missing.append(name + " (invalid format)")
    if missing:
        raise SystemExit("Missing, empty or invalid output files:\n" + "\n".join(missing))
    return sum(i.startswith("tables/") for i in items), sum(i.startswith("figures/") for i in items)


def check_compact(run, reference):
    tables, figures = check_inventory(run, reference)
    for name in REFERENCE_TABLES:
        if not portable_csv_equal(reference / name, run / "tables" / name):
            raise SystemExit(f"{name} differs from the manuscript reference.")
    for name in ("reproduction_audit.csv", "revision_audit.csv"):
        flags = pd.read_csv(run / "tables" / name, dtype=str)["passed"]
        if not flags.eq("True").all():
            raise SystemExit(f"Failed numerical audit in {name}")
    print(
        f"Found {tables} table/matrix files and {figures} figure exports. "
        "Both manuscript tables, both audits (80 checks) and the Figure S4 counts match."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Verify a new run against the manuscript or a full reference directory."
    )
    parser.add_argument("run", nargs="?", type=Path, default=Path("GHIR_reproduction"))
    parser.add_argument(
        "--against",
        type=Path,
        help="Also compare every reference table/matrix against this full results directory",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Allow numeric rounding and text line-ending differences in a full comparison",
    )
    parser.add_argument(
        "--pixels",
        action="store_true",
        help="Also require identical decoded PNG pixels during a portable full comparison",
    )
    args = parser.parse_args()
    reference = Path(__file__).resolve().with_name("reference")
    check_compact(args.run, reference)
    if not args.against:
        return
    paths = sorted((args.against / "tables").glob("*"))
    figures = sorted((args.against / "figures").glob("*.png"))
    expected = (reference / "expected_files.txt").read_text(encoding="utf-8").splitlines()
    expected_tables = {Path(name).name for name in expected if name.startswith("tables/")}
    expected_pngs = {Path(name).name for name in expected if name.endswith(".png")}
    if {p.name for p in paths} != expected_tables or {p.name for p in figures} != expected_pngs:
        parser.error(f"{args.against} does not contain the full reference table and PNG inventory.")
    bad = []
    for archived in paths:
        candidate = args.run / "tables" / archived.name
        if archived.suffix == ".npz":
            same = same_matrix(archived, candidate, args.portable)
        else:
            same = table_bytes(archived) == table_bytes(candidate)
            if not same and args.portable:
                if archived.name.endswith((".csv", ".csv.gz")):
                    same = portable_csv_equal(archived, candidate)
                elif archived.suffix == ".tex":
                    same = archived.read_text(encoding="utf-8") == candidate.read_text(
                        encoding="utf-8"
                    )
        if not same:
            bad.append(candidate)
    compare_pixels = args.pixels or not args.portable
    if compare_pixels:
        for archived in figures:
            candidate = args.run / "figures" / archived.name
            with Image.open(archived) as ref_image, Image.open(candidate) as run_image:
                if not np.array_equal(
                    np.asarray(ref_image.convert("RGBA")), np.asarray(run_image.convert("RGBA"))
                ):
                    bad.append(candidate)
    if bad:
        raise SystemExit("Full comparison differed for:\n" + "\n".join(map(str, bad)))
    print(f"Full reference comparison passed for {len(paths)} table/matrix files.")
    if compare_pixels:
        print(f"Decoded pixels match for all {len(figures)} PNGs.")


if __name__ == "__main__":
    main()
