# CIPHER benchmarks

Forward-prediction benchmarks comparing CIPHER against published perturbation-response
models (GEARS, scGPT, scLAMBDA, scouter, GenePert) and a linear-mean baseline.

Every driver in [`run_files/`](run_files/) takes the **same CLI contract** — no path is
hardcoded, so the benchmarks run anywhere:

```
--splits-dir PATH   root holding <dataset>/{filtered.h5ad,control_idx.npy,train_idx.npy,test_idx.npy}
--dataset NAME ...  one or more dataset names (default: every dataset found under --splits-dir)
--output-dir PATH   results root; each driver writes <output-dir>/<MODEL>/<dataset>/results.pkl
```

Model-specific drivers add `--resources-dir`, `--models-dir`, `--pretrained-dir`,
`--device`, `--epochs`, `--seed`, `--batch-size`, `--skip-existing`/`--overwrite`.
Run any script with `--help` for its exact flags.

## 1. Install

```bash
pip install -e ".[benchmarks]"          # CIPHER + the linear-mean baseline + setup_resources.py
```

The deep-learning baselines each need their own environment (they have mutually
incompatible pins) — see [`conda_envs/`](conda_envs/):

```bash
conda env create -f benchmarks/conda_envs/GEARS.yml     # likewise scGPT / sclambda / scouter / GenePert
```

## 2. Fetch the resource files

Large resources are **downloaded, not committed**:

```bash
python benchmarks/setup_resources.py                    # into benchmarks/resources/
python benchmarks/setup_resources.py --only essential_all_data_pert_genes.pkl
```

The script verifies every file against a known md5 and fails loudly on a mismatch.

| file | size | source | destination | in git? |
| --- | --- | --- | --- | --- |
| `GenePT_gene_protein_embedding_model_3_text.pickle` | 867 MB | Zenodo [10833191](https://zenodo.org/records/10833191) | `--resources-dir` | downloaded |
| `essential_all_data_pert_genes.pkl` | 546 KB | Harvard Dataverse datafile 6934320 (GEARS) | `--resources-dir` | downloaded |
| `gene_alias_to_symbol.pkl` | 647 KB | custom HGNC-derived — **no upstream source** | `--resources-dir` | **committed** |
| `best_model.pt` + `args.json` + `vocab.json` | 196 MB | HF [`wanglab/scGPT-human`](https://huggingface.co/wanglab/scGPT-human) (scGPT authors) | `--models-dir`/`scGPT/pretrained_models/scGPT_human` | downloaded |

Every downloaded file was verified **byte-identical** (md5) to the copies the benchmarks were
validated against. The scGPT URLs are pinned to a commit SHA, so a future push to the mirror
cannot silently swap the weights under the checksum gate; the files land exactly where
`run_scGPT.py --pretrained-dir` looks by default. (Upstream scGPT only publishes a Google Drive
*folder*, which is not directly fetchable — this first-party HF mirror is what makes the
checkpoint scriptable with the standard library alone.)

`gene_alias_to_symbol.pkl` has no upstream artifact and is not byte-reproducible from current
HGNC, so it ships in the repository.

## 3. Prepare the data splits

`dataset_splits/<dataset>/` holds the checked-in `control_idx.npy`, `train_idx.npy` and
`test_idx.npy` for each of the 27 datasets. The matching `filtered.h5ad` is **not**
distributed (it is large and covered by the source datasets' own terms) — place or symlink
it next to the index files:

```bash
ln -s /path/to/<dataset>_filtered.h5ad benchmarks/dataset_splits/<dataset>/filtered.h5ad
```

Indices are positional into that `filtered.h5ad`, so it must be the exact matrix the splits
were generated from. Drivers validate this up front and name any missing file.

## 4. Run

```bash
SPLITS=benchmarks/dataset_splits
OUT=benchmarks/results

# CIPHER (uses the installed `cipher` package)
python benchmarks/run_files/run_CIPHER.py      --splits-dir $SPLITS --output-dir $OUT
# one dataset only
python benchmarks/run_files/run_CIPHER.py      --splits-dir $SPLITS --output-dir $OUT --dataset NormanWeissman2019_filtered

python benchmarks/run_files/run_linear_mean.py --splits-dir $SPLITS --output-dir $OUT

# deep-learning baselines (each in its own conda env, GPU)
python benchmarks/run_files/run_GEARS.py    --splits-dir $SPLITS --output-dir $OUT \
    --resources-dir benchmarks/resources --models-dir benchmarks/models --device cuda
python benchmarks/run_files/run_scGPT.py    --splits-dir $SPLITS --output-dir $OUT \
    --models-dir benchmarks/models --device cuda
python benchmarks/run_files/run_scLAMBDA.py --splits-dir $SPLITS --output-dir $OUT \
    --resources-dir benchmarks/resources --models-dir benchmarks/models --device cuda
python benchmarks/run_files/run_scouter.py  --splits-dir $SPLITS --output-dir $OUT \
    --resources-dir benchmarks/resources --models-dir benchmarks/models --device cuda
python benchmarks/run_files/run_GenePert.py --splits-dir $SPLITS --output-dir $OUT \
    --resources-dir benchmarks/resources --models-dir benchmarks/models
```

Each writes `<output-dir>/<MODEL>/<dataset>/results.pkl` with per-perturbation
`CORRELATION`, `SPEARMAN`, `MSE`, `R2`, `L2`, `COSINE`, plus `DELTA_Y` / `DELTA_X` /
`CONTROL` / `RAW_PREDICTION` profiles and timing / peak-RAM fields.

`run_CIPHER.py` calls the packaged implementation directly — `cipher.compute_covariance`,
`cipher.forward_predict` and `cipher.forward_metrics` — rather than re-implementing the
method, so the benchmark always scores the shipped code.

## 5. Model size — `parameters.csv`

[`parameters.csv`](parameters.csv) records how many parameters each benchmarked model actually
**fits**, so model capacity can be reported alongside accuracy. Counts were derived by
instantiating each vendored architecture with the hyperparameters its driver uses (scGPT was
additionally cross-checked against its pretrained checkpoint), so they are exact.

Because most baselines scale with the number of genes *G*, the CSV carries an exact formula per
model plus a comparable headline number evaluated at **G = 5,000 genes, P = 100 perturbations**:

| model | fitted parameters | fitted by |
| --- | ---: | --- |
| linear_mean | 0 | nothing is fit |
| **CIPHER** | **100** (1 per perturbation) | closed-form least squares |
| GEARS | 1,668,824 | gradient descent |
| scLAMBDA | 9,697,763 | gradient descent |
| GenePert | 15,365,000 | closed-form ridge |
| scouter | 28,002,760 | gradient descent |
| scGPT | 51,859,459 | gradient descent (full fine-tune) |

CIPHER fits **one scalar per perturbation** — ~5×10⁵ times fewer than scGPT. To keep that claim
defensible, the CSV also records, in `estimated_statistics_not_fitted`, what CIPHER *estimates*
without fitting: the control mean (*G* values) and the control covariance Σ (*G(G+1)/2* entries).
These are plug-in sample moments of the control cells — no loss function, no optimiser, no
regularisation — shared across every perturbation, but they are still O(*G*²) numbers derived from
data, so they should be stated rather than glossed over.

## 6. `models/` — vendored third-party source

`models/` holds the baseline model source (GEARS, scGPT, scLAMBDA, scouter, GenePert), ~40 MB,
and **is tracked in this repository**. That is deliberate: every checkout carries local
modifications that upstream does not have (GEARS is a commit ahead of `snap-stanford/GEARS`
plus working-tree edits; scGPT has ~96 modified files), so a plain `git clone` would *not*
reproduce the benchmarked code. See [`models/VENDORED.md`](models/VENDORED.md) for each
model's upstream URL and the exact base commit it was taken from. Licenses remain those of
the upstream projects.

**Pretrained checkpoints are not tracked.** `models/**/pretrained_models/` is gitignored —
scGPT's `best_model.pt` alone is 196 MB and GitHub rejects any file over 100 MB. It is fetched
by `setup_resources.py` instead (see §2), which installs it to:

```
benchmarks/models/scGPT/pretrained_models/scGPT_human/{best_model.pt,args.json,vocab.json}
```

```bash
python benchmarks/setup_resources.py --only best_model.pt --only args.json --only vocab.json
```

That is `run_scGPT.py --pretrained-dir`'s default location; pass the flag to use a copy elsewhere.

Point `--models-dir` at whatever directory holds these checkouts.
