# Reproducing the whole-bone GH and cartilage study

**Projecting a whole-bone growth hormone exposure signature onto embryonic and postnatal cartilage: a test with independent zonal controls**  
Ahmet Haşim Yurttakal and Hasan Erbay

Use this package to reconstruct the signature from public expression data, repeat the embryo and four-week cartilage analyses, and regenerate the two manuscript tables, three main figures and twelve supplementary figures. The notebook explains the theory and shows the results as you run it. The Python scripts perform the same analysis from a terminal.

## Quick start

1. Extract the entire ZIP into a writable folder. Open a terminal in the folder containing `README.md` and `reproduce.py`.
2. Install **64-bit Python 3.12**. Use the commands for your operating system below. Allow about 3 GiB of free disk for a fresh environment, the complete ZIP, extracted files and one new run. An 8 GiB RAM machine is a practical starting point.
3. When JupyterLab opens the notebook, select **Kernel → Restart Kernel and Run All Cells**. It is safe to reuse the verified data cache after restarting. The final section links to your results.

**Already using Anaconda?** Run these commands in Anaconda Prompt from the extracted folder. The environment file pins the same scientific and notebook packages as the pip installation.

~~~text
conda env create -f environment.yml
conda activate ghir-cartilage
python -m jupyterlab GHIR_Public_Notebook.ipynb
~~~

**Windows PowerShell**

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab GHIR_Public_Notebook.ipynb
~~~

**macOS or Linux**

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab GHIR_Public_Notebook.ipynb
~~~

These commands use the environment's interpreter directly. For the examples below, `python` means that same executable. The dependency file includes JupyterLab, the notebook kernel and the scientific packages.

## Choose a download

| Distribution | What it contains | When to use it |
|---|---|---|
| `GHIR_Public_Reproduction.zip` | Notebook, Python source, dependencies specification, all 11 GEO input files, frozen annotations and the full set of reference outputs | Use for an offline data run or comparison of every result. Install the Python environment first. |
| `GHIR_Public_Source.zip` | Same notebook and source, annotations and compact reference checks | Use for a public code repository or a smaller download. The first run retrieves about 352 MiB from GEO. |
| `GHIR_Public_Notebook.ipynb` alone | All analysis functions, two compressed gene-identifier lookups, data URLs, SHA-256 digests, setup and the MIT notice | Use in an existing Jupyter environment. The first notebook section includes a one-command dependency installation. |

The notebook and Python pipeline generate all numerical outputs from expression data. Reference tables and figures are used only by the optional external comparison. The audit cells contain published target values for checking the computed statistics.

## Run without opening Jupyter

Test the installation in seconds, then run the full analysis.

~~~bash
python reproduce.py --check
python reproduce.py
python verify_outputs.py GHIR_reproduction
~~~

For a fresh headless notebook execution using the same interpreter, run `python run_notebook.py`. It saves `GHIR_reproduction/GHIR_executed.ipynb` with the computed tables and inline graphics. It uses an in-process IPython kernel, so a notebook server and a separately registered kernel are unnecessary. The ordinary JupyterLab notebook is also fully standalone.

**Custom folders and offline runs.** The default cache is `inputs/` and results go in `GHIR_reproduction/`. With the complete package you can use `--offline` immediately. The smaller package needs one online run or a copy of the verified cache first. These examples keep the results from the two entry points side by side.

~~~bash
python reproduce.py --download-only
python reproduce.py --offline --output run_python
python run_notebook.py --offline --output run_notebook
python verify_outputs.py run_notebook --against run_python --portable --pixels
~~~

Pass `--inputs "path/to/inputs"` to reuse a cache. The interactive notebook exposes `INPUT_DIRECTORY`, `OUTPUT_DIRECTORY` and `OFFLINE` at the top of step 1. Both entry points also accept `GHIR_INPUTS`, `GHIR_OUTPUT` and `GHIR_OFFLINE=1`. Use different output folders for simultaneous runs.

## Read and verify the results

| Output relative to `GHIR_reproduction/` | What to open |
|---|---|
| `tables/table1.csv` | Original interval membership, excluded lineages and rebuilt embryonic state counts |
| `tables/table2.csv` | Whole-bone and control signatures in both cartilage-dominant atlases. `matched_p` tests interior amplitude. `direction_holm` tests the direction of anatomical controls. |
| `figure_index.csv` and `figures/` | Figure numbers and filenames for all 45 exports. PNGs are 300 dpi, EPS files embed the raster rendering to preserve transparency, and PDFs retain vector text and lines. |
| `tables/H1_figure_values.csv` | Figure S4 exceedance counts and probabilities from the generated draws |
| `tables/reproduction_audit.csv`, `tables/revision_audit.csv` | The 43 historical and 37 revised numerical comparisons with their tolerances and pass flags |
| `environment.json` | Interpreter and package versions recorded during the run |

Three results provide a quick check. Table 1 assigns **782 of 1,833** cells in the historical interval to excluded lineages. The matched interior-amplitude probabilities for the whole-bone projection round to **0.544 at E15** and **0.754 at P28**. The independently learned P28 hypertrophic control increases toward later cells, with **Holm-adjusted directional p = 0.004**. These comparisons concern basal expression in untreated atlases.

`python verify_outputs.py GHIR_reproduction` checks the inventory, graphics headers, both manuscript tables, both audit tables and the Figure S4 values against the compact reference. With the complete package, also run the full numerical comparison.

~~~bash
python verify_outputs.py GHIR_reproduction --against reference_outputs --portable
~~~

This checks all **114 table, sparse-matrix and generated TeX files**. Portable comparisons allow floating-point roundoff (absolute and relative tolerances `1e-8`) and CSV/TeX line-ending differences. Add `--pixels` for equality of all 15 decoded PNGs. Font or renderer differences can affect pixels across operating systems. PDF/EPS creation metadata can vary. The `reference/` folder ships with both editions. The full `reference_outputs/` folder ships with the complete edition.

## Find the code

| File | Purpose |
|---|---|
| `reproduce.py`, `workflow.py` | Command-line entry point and analysis sequence |
| `ghir_methods.py` | Microarray processing, cell graphs, scores, null distributions and implementation checks |
| `revision_analysis.py` | Cartilage selection, independent anatomical controls, sample checks, gene omissions, updated figures and tables |
| `geo_inputs.py`, `run_settings.py` | Verified data downloads, cache handling, version checks and paths |
| `run_notebook.py`, `verify_outputs.py` | Headless notebook execution and comparison with reference results |
| `requirements.txt`, `requirements-linux-py312.lock` | Cross-platform direct pins and a complete snapshot of the tested Linux/Python 3.12 environment |
| `DATA.md`, `input_manifest.tsv`, `annotations/` | Accession records, source descriptions, file sizes, SHA-256 values and frozen lookups |
| `prepare_annotation.py` | Optional provenance utility to rebuild the frozen MGI subset given the matching full report |
| `CITATION.cff`, `LICENSE` | Citation metadata and the MIT software license |

## If a run stops

- **A package is missing or Jupyter opens the wrong environment.** Run the installation command above using the same interpreter that launches Jupyter. In a notebook, `import sys; print(sys.executable)` shows the active interpreter. Restart the kernel after installing packages.
- **GEO is unreachable or a checksum fails.** Run `python reproduce.py --download-only` to fetch or repair the cache. With restricted internet access, copy `inputs/` from the complete package and use `--offline`. A verified cache may be read-only. The output folder must be writable.
- **The run stops on a numerical check.** Inspect the quantity, computed value, target and tolerance in the relevant audit CSV, along with `environment.json`. Rerun in a new directory with the pinned environment. Retain the seed, null-draw count and tolerance when comparing to the manuscript.
- **The machine runs low on memory.** Close other analysis processes and run one entry point at a time. The postnatal matrix contains about 45.6 million nonzero entries. Streaming the companion matrix and using sparse counts reduce peak memory.
- **A long step appears quiet.** The null distributions and the cell-by-cell postnatal format comparison take time. Headless notebook execution prints each step. The interactive notebook shows a busy kernel during computation.

The code and accompanying documentation are released under the MIT License. The original GEO, MGI and Ensembl records identify the producers of the deposited data and annotations. [Data provenance](DATA.md) · [Software citation](CITATION.cff) · [License](LICENSE)
