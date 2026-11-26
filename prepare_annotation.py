"""Reconstruct the frozen GPL922 MGI subset from the specified release of the full report."""

import argparse
import csv
import hashlib
from pathlib import Path
import pandas as pd
from ghir_methods import CANONICAL_GH_IGF, parse_geo_family_soft, platform_probe_to_gene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mgi_report", type=Path, help="MRK_List2.rpt, the exact frozen full report")
    p.add_argument("--platform", type=Path, default=Path("inputs/GSE3689_family.soft.gz"))
    p.add_argument("--out", type=Path, default=Path("mgi_gpl922.tsv"))
    args = p.parse_args()
    with args.mgi_report.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != "b5ca6f34bef6f3043eaf0e1d2a3da813f34ce676ab3d87b30887353862c55cdc":
        raise SystemExit(
            "The input is a different MGI release. Use the included frozen subset to reproduce the paper."
        )
    geo = parse_geo_family_soft(args.platform)
    targets = (
        set(platform_probe_to_gene(geo["platforms"]["GPL922"]["table"]).str.upper())
        | CANONICAL_GH_IGF
    )
    entries = []
    with args.mgi_report.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            symbol = record["Marker Symbol"].upper()
            aliases = {
                a.strip().upper()
                for a in record["Marker Synonyms (pipe-separated)"].split("|")
                if a.strip()
            }
            for label in targets & (aliases | {symbol}):
                entries.append(
                    (
                        label,
                        symbol,
                        record["MGI Accession ID"],
                        "official" if label == symbol else "alias",
                    )
                )
    table = pd.DataFrame(sorted(entries), columns=["array_label", "official", "mgi_id", "match"])
    table.to_csv(args.out, sep="\t", index=False, lineterminator="\n")
    if (
        hashlib.sha256(args.out.read_bytes()).hexdigest()
        != "966575a12e72f61c7d5a23f3844ee2f2f11d1fa22ef044c0ec58597de4d9a05f"
    ):
        raise RuntimeError("Annotation output differs from the reference snapshot.")


if __name__ == "__main__":
    main()
