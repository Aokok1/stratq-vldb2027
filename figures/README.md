# Evaluation figures

Run `python experiments/plot_sprint3_figures.py` from the project root to regenerate the five manuscript figures as vector PDFs and 300 dpi PNG previews. The [Sprint 3 metrics manifest](figure_metrics_sprint3.json) records source JSON hashes and plotted values. The renderer checks text bounds and overlaps where applicable.

## Manuscript figures

- [Fig. 1 architecture](fig1_architecture.pdf): DuckDB/Arrow profiling, predictable Safe-LOO gate, outward-rounded WoR CS, and certified query outputs.
- [Fig. 2 GPU workloads](fig2_gpu_workloads.pdf): RTX 4060 exact/fresh/reused latency after proxy scoring, COUNT calls, and TOP-2 certification. Baseline decision bars use the post-hoc COUNT-derived protocol.
- [Fig. 3 proxy sweep and Zipf](fig3_proxy_sweep_zipf.pdf): proxy AUC sweep and skewed group scaling. Coverage labels aggregate all 15 `(G, alpha)` configurations (two seeds each).
- [Fig. 4 Safe-LOO and Adult budgets](fig4_sprint1_adult.pdf): matched ablation and Adult fixed-budget operators. The six comparison operators are fixed-look/asymptotic adaptations, not anytime validity tests.
- [Fig. 5 scale and memory](fig5_scale_memory.pdf): profile amortization and isolated 10M subprocess memory. `Q=10` is extrapolated beyond five measured queries; the query subprocess includes profile rebuild. The +21.7 MB value is one immediate pre-teardown RSS delta, the query traced-allocation peak is 85.6 MB, and the final post-teardown delta is −0.68 MB.

The older three-figure set (`proxy_calls`, `adult_fixed_budget`, `parquet_scale`) remains for historical review but is not the 12-page manuscript's figure set.
