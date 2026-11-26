# StratQ reproducibility package — VLDB 2027 RC1 candidate

This is a clean local candidate for GitHub/Zenodo, not a public deposit or an evaluated artifact. It contains code, tests, experiment scripts, frozen result JSON, figure sources/PDFs/PNGs, the 12-page manuscript source/PDF, proof/audit records, and a SHA-256 manifest. It excludes datasets, model weights, caches, `__pycache__`, `.pytest_cache`, and raw 10M-row Parquet snapshots.

No badge, public repository URL, author identity, or license grant is claimed. The authors must confirm the code license and publish the final archive before filling `\vldbavailabilityurl{}`.

## 1. Audit the frozen measurements offline

A CPU suffices for the core audit; no GPU, network, model credentials, or database server is required. From the extracted root:

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

The claim audit reads the included `stratq.pdf`, `stratq.bbl`, and `stratq.log`; it does not require TeX. It checks raw/summary JSON, Sprint 4 descriptive statistics, proof/code consistency, bibliography structure, CMT consistency, 12-page body limit, and LaTeX hard gates. The online bibliography evidence is in `BIBLIOGRAPHY_AUDIT.md`; the offline script is a structural regression, not a replacement for online verification.

To regenerate figures from frozen JSON:

```powershell
& $py experiments/plot_sprint3_figures.py
```

To rebuild the manuscript, enter `4_LaTeX_Source/vldb_template` and run `latexmk -g -pdf -interaction=nonstopmode -halt-on-error stratq.tex` with TeX Live 2025. The official PVLDB template files are included.

## 2. Fetch checksum-pinned data and models

Use a separate extraction for reruns because scripts overwrite their result JSON. The default command fetches all core and GPU inputs:

```powershell
& $py artifact_tools.py fetch-data
```

For CPU-only experiments:

```powershell
& $py artifact_tools.py fetch-data --core-only
```

Downloads are streamed to a temporary `.part` file, checked for exact size and SHA-256, and atomically renamed only after verification. Existing mismatches are rejected rather than overwritten. Hugging Face inputs have fixed repository revisions and a checksum-equivalent mirror fallback.

| Input | Source / revision | SHA-256 |
|---|---|---|
| Adult archive | [UCI Adult](https://doi.org/10.24432/C5XW20) | `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` |
| SMS Spam archive | [UCI SMS Spam](https://doi.org/10.24432/C5CC84) | `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3` |
| Fashion-MNIST four IDX gzip files | [TensorFlow mirror](https://storage.googleapis.com/tensorflow/tf-keras-datasets/) | Hashes are in `results/fashion_live.json` and enforced by `fetch-data`. |
| Yelp test Parquet | [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full), revision `c1f9ee939b7d05667af864ee1cb066393154bf85` | `bf06d5969bff93ecd4a6d4b330643761ce108b42e9b8a894cf82c9df02a08540` |
| CIFAR-10 Python archive | [Toronto](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |
| ResNet34 ImageNet backbone | [PyTorch weights](https://download.pytorch.org/models/resnet34-b627a593.pth) | `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f` |
| DistilBERT-SST2 weights | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), revision `714eb0fa89d2f80546fda750413ed43d93601a13` | `7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa` |
| DistilBERT `config.json` | same revision | `582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15` |
| DistilBERT `tokenizer_config.json` | same revision | `5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0` |
| DistilBERT `vocab.txt` | same revision | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

Data/model licenses remain those of their upstream owners. Yelp Review Full is derived from the Yelp Dataset Challenge and carries its own dataset terms; CIFAR-10 and Fashion-MNIST have their upstream attribution terms; DistilBERT is Apache-2.0; the torchvision ResNet34 backbone is distributed by PyTorch. No third-party data or weights are redistributed in this ZIP. Authors must verify the final repository's license compatibility before public deposit.

## 3. Rerun map

Run scripts from the extracted root. Timing results are host-specific; compare non-timing outputs and call/coverage semantics rather than exact milliseconds.

| Evidence | Scripts, in order | Comparison |
|---|---|---|
| Core offline audits | `artifact_tools.py verify`; `pytest -q`; three Sprint 4 audit scripts | Manifest, unit tests, bibliography/CMT/claim gates |
| Adult replay and classical baselines | `adult_count_replay`, `adult_hypergeom`, `adult_fixed_budget`, `adult_blazeit`, `adult_senate`, `adult_verdict`, `adult_online_aggregation`, `adult_lws`, `adult_routing`, `adult_routing_capped` | Exact totals, coverage, widths, calls |
| SMS/Fashion | `sms_live`, `fashion_live`, `audit_sms`, `audit_fashion` | Held-out group counts and query outcomes |
| Safe-LOO ablation | `sprint1_safe_gating_ablation` | Paired calls, widths, coverage; SMS legacy is included |
| CPU timing | `sprint4_cpu_latency` | Aligned exact/fresh/reused protocol; supersedes cross-study timing comparison |
| Synthetic scale | `scale_count_replay.py --rows N` for 100000/1000000/10000000, then `audit_scale` | Coverage, width, calls, timings |
| Decision boundaries | `decision_boundaries`, `audit_decisions` | Wrong certificates, abstentions, tie census |
| Sprint 2 systems | `sprint2_systems` | Proxy sweep, Zipf, amortization, isolated memory stages |
| Sprint 2 GPU | `sprint2_gpu_workloads` after installing optional GPU dependencies | Reference counts, decisions, latency, VRAM |
| Yelp mechanism diagnostic | `sprint4_yelp_gate_diagnostic --device cuda` | Safe-LOO trace; rejects range-penalty explanation |
| Derived Sprint 4 statistics | `sprint4_statistics` | SD/ranges/denominators from frozen JSON |
| Figures | `plot_sprint3_figures` | Vector PDFs, PNG previews, metrics manifest |

Optional GPU environment used for the recorded run: Python 3.12.6, PyTorch 2.8.0+cu129, torchvision 0.23.0+cu129, CUDA 12.9, transformers 5.2.0, RTX 4060 Laptop 8 GiB. Install the matching PyTorch/torchvision CUDA wheels from the official PyTorch selector, then `pip install transformers==5.2.0`. The core test suite does not import Torch.

The CIFAR-10 reference model is fine-tuned for two epochs in the runner. Backbone/data hashes are pinned, but GPU/bf16 training is not guaranteed bitwise identical across hardware; the saved fine-tuned model hash in the frozen JSON is a measurement fingerprint, not a cross-hardware reproducibility promise. The Yelp DistilBERT reference is a fixed upstream model.

## 4. Provenance and scope boundaries

- Frozen result JSON retain the code hashes under which they were produced. Sprint 4 hardened normalized-label enclosure and the GPU harness without rewriting measurements. `results/sprint4_code_change_validation.json` records an Adult replay with identical calls/decisions and only outward endpoint widening (maximum about $2.9\times10^{-11}$ count units).
- `results/sprint4_summary_statistics.json` recomputes sample SD, ranges, coverage denominators, latency setup boundaries, and memory definitions from raw JSON.
- Modern baselines are operator-level adaptations, not full system replicas. BARGAIN-Global is a global-count surrogate; PilotDB-Asymp optional stopping is a fixed-look CLT diagnostic; Spertus-WoR does not implement UI-TS; BlazeIt is an inspired fixed-look arm.
- Three-seed GPU/proxy results and two-seed Zipf cells are descriptive. Observed coverage is not a population coverage proof.
- GPU latency uses precomputed proxy scores and warm model/OS caches. ``Fresh'' rebuilds the TableProfile; proxy fitting/scanning is excluded and reported separately.
- The 10M memory study is offline Boolean-label replay. The +21.69 MB value is one immediate pre-teardown RSS delta; the query traced-allocation peak is 85.6 MB and the final post-teardown delta is −0.68 MB.

## 5. Release status

The ZIP is a local RC1 candidate. Public GitHub/Zenodo upload, license confirmation, DOI/URL registration, and CMT submission have not been performed. See `SUBMISSION_PREFLIGHT.md` for the four author-controlled switches.
