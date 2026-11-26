# StratQ experiments

## Repeated online latency

After setting up both live datasets below, run `python experiments/live_latency.py` and `python experiments/audit_latency.py`. The timed requests start with fitted CPU models and input rows in memory. Fresh-profile requests include group assignment, Arrow conversion, DuckDB profiling, and, in the proxy arm, scoring. The reuse arm excludes setup. Exact requests include group assignment, batched reference inference, and exact aggregation. Each workload has one warm-up block and ten measured blocks, each with randomized method order. Exact timing is reported at the fastest median among the tested batch sizes, not as an estimate of the best achievable performance. Model fitting and data loading are excluded. [Raw timings and bounds](../results/live_latency.json)

## Fashion-MNIST live inference

Download the four official IDX gzip files from the [TensorFlow mirror](https://storage.googleapis.com/tensorflow/tf-keras-datasets/) to `data/fashion/`. The file names and SHA-256 hashes are recorded in [fashion_live.json](../results/fashion_live.json). Run `python experiments/fashion_live.py` and verify the saved result with `python experiments/audit_fashion.py`.

The query counts footwear predictions (sandals, sneakers, and ankle boots) on the 10,000 official test images. Four cut points estimated from training-set pixel means define five brightness groups. Both models use the same 6,000-image stratified training subset. A logistic regression on pooled pixels supplies the proxy. An RBF SVM on full-resolution pixels supplies the on-demand reference predicate. The CPU run uses five matched query seeds. Each group targets a count interval half-width of 4% of its size at nominal 90% simultaneous coverage.

Query times reuse the profile. The result file records proxy scoring, profile construction, and one exhaustive inference scan at each of three batch sizes. Exhaustive scans use the same fitted reference model and held-out images. [Fashion-MNIST dataset and attribution](https://github.com/zalandoresearch/fashion-mnist). The upstream repository provides the [MIT License](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE), copyright 2017 Zalando SE.

## SMS live inference

Download the [UCI archive](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip) as `data/uci_sms.zip` and run `python experiments/sms_live.py` with scikit-learn 1.8.0. Train a 256-feature word TF–IDF logistic model as the proxy and a character TF–IDF five-nearest-neighbor model as the reference. Split the corpus by normalized text identity to keep duplicates on the same side of the split. Group held-out messages by character length with cuts at 40, 80, 120, and 160. Count agreement is measured against the full reference prediction on this fixed snapshot. Compare `COUNT`, `HAVING count >= 40`, and `TOP-2` under ten matched seeds. `COUNT` targets a half-width of six percent of the group size.

Query wall time reuses the constructed profile. The first proxy scan and profile construction are reported separately. Exhaustive model timings use three batch sizes with one observed run per size. The fitted models use corpus annotations from the training split. Per-query outputs, exhaustive counts, and timing records are in [sms_live.json](../results/sms_live.json).

Dataset attribution follows the UCI repository record for Tiago Almeida and José Hidalgo, *SMS Spam Collection* (2011), [DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84), CC BY 4.0.

## Adult grouped count

From the project root, download the [official UCI archive](https://archive.ics.uci.edu/static/public/2/adult.zip) as `data/adult_uci.zip` and run `python experiments/adult_count_replay.py`. The recorded run used Python 3.12.6, DuckDB 2.0.1, NumPy 2.4.1, and PyArrow 21.0.0. The script writes the Arrow data to `data/adult_test_replay.parquet` and records each run in [adult_count_replay.json](../results/adult_count_replay.json).

The query counts incomes above $50,000 by workclass on the 16,281-row official test split. The `?` workclass is retained as a separate group. Ground truth comes from a full DuckDB aggregation. The proxy is the Laplace-smoothed positive rate by education level estimated on the official training split. The covariates variant includes age and weekly hours. All variants use matched seeds, the same batch configuration, and a count error target of 4% of each group size. The results are for label replay. The dataset represents historical Census records and serves here as a fixed query snapshot.

Dataset attribution: Barry Becker and Ronny Kohavi, *Adult* (1996), UCI Machine Learning Repository, [DOI 10.24432/C5XW20](https://doi.org/10.24432/C5XW20), CC BY 4.0.

## Parquet scale benchmark

Run `python experiments/scale_count_replay.py --rows N` in a fresh Python process for each of `100000`, `1000000`, and `10000000`. The generator uses 20 Zipf-skewed groups and a noisy proxy derived from the latent variable used to generate the binary label. Each scale uses three paired query seeds. Independent generator counters and a full DuckDB aggregation cross-check the exact counts.

The results report Parquet generation time, full-label SQL aggregation time, StratQ profile construction time, and query time with the existing profile. Query time includes per-group permutation generation. The SQL aggregation precedes profiling and may warm the file cache. Peak working set is a Windows process-lifetime measurement that includes generation, SQL verification, profiling, and query execution. Reference labels are replayed from memory. Run `python experiments/audit_scale.py` to recompute coverage, width attainment, paired call savings, and query-time medians from the saved records. See the [100K](../results/scale_100000.json), [1M](../results/scale_1000000.json), and [10M](../results/scale_10000000.json) records.

## Sprint 2 GPU workloads

Run `python artifact_tools.py fetch-data` from the project root to obtain the revision-pinned Yelp test Parquet, CIFAR-10 archive, DistilBERT-SST2 files, and ResNet34 backbone with SHA-256 verification. Install the optional CUDA PyTorch/torchvision/transformers stack documented in [`ARTIFACT.md`](../ARTIFACT.md), then run `python experiments/sprint2_gpu_workloads.py`. The runner uses an RTX 4060 in the recorded experiment; another GPU changes timings and may change the fine-tuned CIFAR reference outputs.

Yelp uses 18,000 disjoint proxy-training rows and 20,000 evaluation rows from the official test Parquet. CIFAR-10 fine-tunes ImageNet ResNet34 for two epochs at 224×224 and detects six animal classes. COUNT uses three matched seeds. HAVING/TOP-2 use dedicated StratQ routing, while modern baseline decisions are derived post hoc from COUNT intervals; this is not an equal-budget decision-routing comparison.

Latency uses precomputed proxy scores and warm model/OS caches. Fresh means a new TableProfile; the frozen run shared the process-local psi cache. The current runner clears that cache before fresh-profile repeats and records the 3/4/5% tolerance schedule. Proxy training/scanning is excluded from headline latency and reported separately.

## Sprint 4 audits and lightweight diagnostics

- `python experiments/sprint4_statistics.py` recomputes sample SD/ranges, coverage denominators, timing boundaries, and memory definitions from frozen JSON.
- `python experiments/sprint4_yelp_gate_diagnostic.py --device cuda` performs one Yelp reference pass and traces Safe-LOO decisions for the three COUNT seeds. It verifies frozen group totals and finds negligible prediction range, rejecting the earlier range-penalty explanation for Yelp's call overhead.
- `python experiments/sprint4_cpu_latency.py` runs the aligned exact/fresh/reused CPU protocol and supersedes the older cross-study timing comparison.
- `python experiments/audit_bibliography_sprint4.py`, `audit_submission_sprint4.py`, and `audit_paper_claims_sprint3.py` are offline hard gates. The manuscript audit includes PDF/BibTeX/LaTeX checks and should be run after compilation.
