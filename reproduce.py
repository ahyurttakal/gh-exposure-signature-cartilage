"""Reproduce the whole-bone GH exposure and cartilage analysis."""

import argparse
from pathlib import Path

from geo_inputs import fetch_geo_inputs, geo_records
from run_settings import check_environment, display_path, offline_mode, resolve_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, help="GEO cache directory (default: inputs/)")
    parser.add_argument(
        "--output", type=Path, help="Results directory (default: GHIR_reproduction/)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="Read checksummed local GEO files without contacting the archive",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and run six small numerical tests, without downloading data",
    )
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Prepare or verify the input cache and exit; uses only the Python standard library",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    inputs, output = resolve_paths(project, args.inputs, args.output)
    if args.check:
        check_environment()
        from ghir_methods import run_regression_checks

        run_regression_checks()
        print("Environment and six implementation checks passed.")
        return
    versions = None if args.download_only else check_environment()
    records = geo_records((project / "input_manifest.tsv").read_text(encoding="utf-8"))
    fetch_geo_inputs(inputs, records, offline_mode(args.offline))
    if args.download_only:
        return
    from workflow import run_analysis

    print(f"Writing results to {display_path(output)}", flush=True)
    run_analysis(inputs, output, versions)


if __name__ == "__main__":
    main()
