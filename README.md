# StratQ — artifact for *Anytime-Valid Grouped Query Contracts for Expensive Predicates*

Reproducibility package for the PVLDB Vol. 20 (VLDB 2027) submission
*StratQ: Anytime-Valid Grouped Query Contracts for Expensive Predicates*.

StratQ is a query execution engine and statistical contract for fixed-snapshot
grouped queries with expensive binary predicates. It couples remaining-population
centering with a predictable, analytically cross-validated Ridge control variate,
computes all group intervals through directed binary64 outward rounding, and shares
a single simultaneous coverage event across all groups and data-dependent reveal
times. That one event directly certifies `COUNT` tolerances, `HAVING` thresholds,
and top-`k` membership, while still allowing explicit abstention under budget caps.
The engine runs over DuckDB and Apache Arrow.

This repository holds **code, tests, experiment scripts, frozen result JSON, figure
sources, the manuscript source and PDF, and the audit records**. It deliberately
does **not** redistribute datasets or model weights: every input is fetched on
demand and checked against a pinned SHA-256 by `artifact_tools.py`. See
[ARTIFACT.md](ARTIFACT.md) for the full input table, provenance, and scope boundaries.

## What is here

| Path | Contents |
|---|---|
| `src/stratq/` | Engine: sampling, control variates, simultaneous confidence sequences, decision certification |
| `tests/` | 52 unit tests plus 3 subtests over the estimator, certificates, gating, and baselines |
| `experiments/` | Rerun scripts for every table and figure, plus the claim / bibliography / submission audits |
| `results/` | Frozen result JSON for all reported measurements |
| `figures/` | Vector PDFs, PNG previews, and the metrics manifest |
| `4_LaTeX_Source/vldb_template/` | Manuscript source, bibliography, compiled PDF, and the official PVLDB template files |
| `ARTIFACT_SHA256.json` | SHA-256 manifest over all 134 artifact payload files |

`README.md`, `LICENSE`, and `.gitattributes` were added when publishing and are
therefore outside the manifest, which pins the RC1 payload set only.

Audit records: [PROOF_PACKAGE.md](PROOF_PACKAGE.md), [PROOF_AUDIT.md](PROOF_AUDIT.md),
[BASELINE_AUDIT.md](BASELINE_AUDIT.md), [BIBLIOGRAPHY_AUDIT.md](BIBLIOGRAPHY_AUDIT.md),
[NOVELTY_AUDIT.md](NOVELTY_AUDIT.md), [VLDB_PC_REVIEW_SPRINT4.md](VLDB_PC_REVIEW_SPRINT4.md),
[SPRINT4_REPORT.md](SPRINT4_REPORT.md), [ARTIFACT_RERUN.md](ARTIFACT_RERUN.md).

## Quick start — audit the frozen measurements

A CPU is enough for the core audit. No GPU, network access, model credentials, or
database server is required. From the repository root:

```powershell
python -m venv .venv
$py = '.venv/Scripts/python.exe'
& $py -m pip install --only-binary=:all: -r requirements-artifact.txt
& $py -m pip check
& $py artifact_tools.py verify
& $py -m pytest -q
& $py experiments/audit_bibliography_sprint4.py
& $py experiments/audit_submission_sprint4.py
& $py experiments/audit_paper_claims_sprint3.py
```

`artifact_tools.py verify` checks every file against the SHA-256 manifest. The claim
audit reads the included `stratq.pdf`, `stratq.bbl`, and `stratq.log`, so it needs no
TeX installation. It cross-checks raw and summary JSON, the Sprint 4 descriptive
statistics, proof/code consistency, bibliography structure, CMT consistency, the
12-page body limit, and the LaTeX hard gates.

## Quick start — rerun experiments

Use a separate checkout for reruns, because scripts overwrite their result JSON.

```powershell
& $py artifact_tools.py fetch-data              # all core and GPU inputs
& $py artifact_tools.py fetch-data --core-only  # CPU-only experiments
```

Downloads stream to a temporary `.part` file, are checked for exact size and SHA-256,
and are renamed atomically only after verification. Hugging Face inputs use fixed
repository revisions with a checksum-equivalent mirror fallback. Data and model
licenses remain those of their upstream owners.

To regenerate the figures from the frozen JSON:

```powershell
& $py experiments/plot_sprint3_figures.py
```

To rebuild the manuscript, enter `4_LaTeX_Source/vldb_template` and run
`latexmk -g -pdf -interaction=nonstopmode -halt-on-error stratq.tex` with TeX Live 2025.

The recorded GPU environment was Python 3.12.6, PyTorch 2.8.0+cu129,
torchvision 0.23.0+cu129, CUDA 12.9, transformers 5.2.0, on an RTX 4060 Laptop 8 GiB.
Install matching PyTorch/torchvision CUDA wheels from the official selector, then
`pip install transformers==5.2.0`. The core test suite does not import Torch.

## Scope and honesty boundaries

These are stated in the paper and repeated here so that a reader of the code alone
does not over-read it:

- Modern baselines are **operator-level adaptations, not full system replicas**.
  BARGAIN-Global is a global-count surrogate; PilotDB-Asymp optional stopping is a
  fixed-look CLT diagnostic; the decision baselines are post-hoc COUNT-derived.
- Three-seed GPU/proxy results and two-seed Zipf cells are **descriptive**. Observed
  coverage is not a population coverage proof.
- GPU headline latency uses precomputed proxy scores and warm model/OS caches.
  "Fresh" rebuilds the `TableProfile`; proxy fitting and scanning are excluded from
  the headline and reported separately.
- The 10M-row memory study is offline Boolean-label replay. `+21.69 MB` is a single
  immediate pre-teardown RSS delta; the query traced-allocation peak is 85.6 MB and
  the final post-teardown delta is −0.68 MB.
- Yelp `COUNT` is a **negative result**: 4,349 calls versus 3,859 for the Spertus-WoR
  arm. The gate trace shows a negligible prediction range, so the earlier
  range-penalty explanation was withdrawn.
- Frozen result JSON retain the code hashes under which they were produced.
  Sprint 4 code changes preserve Adult call counts and decisions and only widen
  interval endpoints outward (maximum about 2.9×10⁻¹¹ count units); see
  `results/sprint4_code_change_validation.json`.
- GPU/bf16 fine-tuning is not guaranteed bitwise identical across hardware. The saved
  fine-tuned model hash in the frozen JSON is a measurement fingerprint, not a
  cross-hardware reproducibility promise.

## Citation

```bibtex
@misc{stratq2027artifact,
  title  = {StratQ: Anytime-Valid Grouped Query Contracts for Expensive Predicates --- Artifact},
  year   = {2026},
  note   = {Reproducibility package submitted with the PVLDB Vol.~20 manuscript},
  url    = {https://github.com/Aokok1/stratq-vldb2027}
}
```

The author block and final availability URL in the manuscript are filled in at
submission time; the manuscript source in this repository keeps the placeholder
required by the review process.

## License

Code in this repository is released under the [Apache License 2.0](LICENSE).
The PVLDB `acmart.cls`, `pvldb.sty`, and `ACM-Reference-Format.bst` template files
retain their upstream terms. Datasets and model weights are **not** included and
remain under their own upstream licenses.
