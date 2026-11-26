# Data and reference files

The release reconstructs the expression measurements from six public GEO series. The two single-cell series each contribute one deposited library. All 11 downloaded GEO files have fixed byte counts and SHA-256 digests in [input_manifest.tsv](input_manifest.tsv).

| Dataset | Material and role | Source record |
|---|---|---|
| GSE3689 | Whole femoral/tibial bone from four-week-old GH-deficient mice after GH or PBS. Six fixed-dye arrays define the source weights. | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE3689) |
| GSE271634 | E15.0 mouse limb cells, sample GSM8380938. The Matrix Market counts, features and barcodes provide the embryonic atlas. | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271634) |
| GSE288529 | Four-week mouse epiphysis, GSM8769462. The HDF5 count matrix and its three companion Matrix Market files provide the postnatal comparator and an independent file-format check. | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288529) |
| GSE87605 | Newborn mouse zonal dissections. These measurements train the resting and hypertrophic control programs. | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87605) |
| GSE14007 | Day-9 mouse zonal dissections. Each animal contributed all zones, allowing an independent anatomical check. | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14007) |
| GSE9160 | Distal femoral growth-plate zones from two human donors, used for the exploratory cross-species comparison. | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9160) |

The source records identify each depositor and the associated studies. Consult those records for upstream terms and cite the original datasets when reusing them.

## Frozen gene-identifier lookups

- `annotations/mgi_gpl922.tsv.gz` contains the historical mouse marker names used for GPL922. The pairings come from the Mouse Genome Informatics marker report. The uncompressed analysis slice has SHA-256 `966575a12e72f61c7d5a23f3844ee2f2f11d1fa22ef044c0ec58597de4d9a05f`. The exact full report used to prepare it has SHA-256 `b5ca6f34bef6f3043eaf0e1d2a3da813f34ce676ab3d87b30887353862c55cdc`. The report at the public URL is updated over time. [MGI marker report](https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt)
- `annotations/ortholog_pairs.csv.gz` contains 13,680 frozen Ensembl BioMart/Compara one-to-one mouse–human gene-identifier pairs used in the study. The uncompressed CSV has SHA-256 `dd584b960ebde57bf64c5aa55caa96c95e09835385b03db00ebac2fc35b79e21`. Both analysis routes use the identical snapshot. [Ensembl comparative genomics](https://www.ensembl.org/info/genome/compara/homology_method.html)

The standalone notebook embeds both lookups, checks their uncompressed hashes, and reads them in memory. The Python route uses the copies in `annotations/`. Lookup files contain identifiers. Observed matrices, fitted statistics, simulations and plots are regenerated from the expression data.

## How the reference outputs were obtained

`reference/` contains the two manuscript tables, the Figure S4 summaries, both numerical audit tables, a figure index and the expected file inventory extracted from the manuscript revision. The complete distribution also includes the full archived tables and figures in `reference_outputs/`. The publication checks use these frozen values as independent comparison targets. They are not inputs to fitting, ranking or figure generation.

The final step creates a fresh figure index from the files just written. Date fields in EPS and PDF can change at each run. The full verifier compares table contents, the sparse numerical matrix and, when requested, decoded PNG pixels.

## Attribution and upstream notices

Mouse Genome Informatics, The Jackson Laboratory, licenses its data and annotations under CC BY 4.0. This package adapts the marker report by selecting GPL922-associated records and the tested panel, normalizing symbol case, splitting synonym lists and recording each historical-to-approved mapping. These changes were made for this analysis. [MGI licensing notice](https://www.informatics.jax.org/mgihome/other/copyright.shtml) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The six expression studies retain the attribution in their GEO series and sample records. The Ensembl identifiers retain attribution to Ensembl and its contributors. [GEO distribution notice](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html) · [Ensembl data notice](https://www.ensembl.org/about/articles/ensembl-content-disclaimer) · [EMBL-EBI terms](https://www.ebi.ac.uk/about/terms-of-use/)
