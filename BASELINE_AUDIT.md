# StratQ baseline audit

*Updated 2026-11-26.* Six operator-level adaptations have been benchmarked; none is a complete system replica. ShadowAQP and Sample + Seek are citation-only at present: the ShadowAQP publisher PDF download did not finish, the Sample + Seek ACM full-text endpoint returned 403, and the archive mirror could not be read. LOTUS is a candidate for a separate filter-then-count comparison, but its filter calibration targets row-level precision and recall, not simultaneous group count accuracy. None of these candidates is currently counted.

| Comparison | Current implementation | Reporting status |
|---|---|---|
| Exhaustive reference | Full prediction pass and exact aggregation | Reference for accuracy and measured inference cost. The text and vision runners execute the reference model directly. |
| StratQ with proxy disabled | Same inference and stopping code as StratQ, with the predictor disabled | Internal ablation. |
| Grouped hypergeometric confidence sequence | Independent binary specialization of the prior–posterior ratio of Waudby-Smith and Ramdas | Verified on small populations and evaluated on Adult. Published source is NeurIPS 2020. |
| Uniform-Strat | Equal allocation delegates to StratQ. Proportional mode uses a fixed-budget draw | Internal sampling comparisons. BlinkDB fidelity is unverified. |
| ABAE-Global | Local importance-weighted residual heuristic with a global stopping rule | Fidelity and sequential coverage are unverified. |
| ABAE-Group | Delegates to grouped StratQ with proxy-only controls | Internal ablation. |
| Senate allocation + fixed-look intervals | Equal-quota subgroup sampling following Congressional Sampling §4.4; local census caps, redistribution, and fixed-look intervals | Standalone SIGMOD-derived operator evaluated on Adult; not the multi-grouping Congress sample or a system replica. |
| SUPG and LOTUS adapters | Selection and cascade objectives | The grouped count contract requires a separately justified adaptation. |

## Online Aggregation fixed-look randomized scan

[Online Aggregation (SIGMOD 1997), §4 and Eq. (1)](https://www.comp.nus.edu.sg/~tankl/cs5208/readings/online.pdf) uses randomized retrieval and conservative Hoeffding intervals for running means over bounded values. The benchmark applies that scan path to binary labels, converting each group's mean to a count by multiplying by its known size. At a predeclared global budget, conditional on the group quotas, each sampled group prefix is uniform without replacement. Equal risk splitting and deterministic feasibility clipping are local additions; an unsampled group gets its full feasible interval. We do not implement the paper's group-prioritized index path or claim optional-stopping validity. At 2,048 / 4,096 / 8,192 calls the mean largest group rate half-widths are 47.62 / 44.52 / 33.17 percentage points, all 90 trials cover, and 21 / 15 / 5 of 30 trials leave an unsampled group. [Raw results](results/adult_online_aggregation.json) · [Implementation](src/stratq/online_aggregation.py) · [Tests](tests/test_online_aggregation.py)

## Senate fixed-budget comparison

[Congressional Sampling (SIGMOD 2000), §4.4 and footnote 13](https://sigmodrecord.org/?smd_process_download=1&download_id=6664) specify uniform random samples of equal size for each nonempty group of a fixed grouping, simplifying the presentation by assuming each group is large enough. This implementation caps quotas at group size, redistributes unused calls, and rounds ties in stable group order. Equal-tailed hypergeometric inversion and Bonferroni correction are additions for this experiment. They cover at the one predeclared look in real arithmetic, not simultaneously over repeated looks. Rational inversion tests cover every possible observed count for populations up to nine.

At the 2,048 / 4,096 / 8,192 call budgets, the mean largest per-group rate half-width is 6.22 / 4.19 / 1.60 percentage points, and observed simultaneous coverage is 26/30, 29/30, and 28/30. The corresponding StratQ widths are 8.82 / 5.29 / 2.73 points. This shows the advantage of the fixed-look comparator on Adult; the comparison changes both allocation and inference. [Raw results](results/adult_senate.json) · [Runner](experiments/adult_senate.py) · [Tests](tests/test_senate.py)

## ABae-inspired fixed-budget comparison

[ABae, Sections 3.1, 3.2, and 4.5](https://arxiv.org/html/2108.06313v1), uses proxy quantile strata, two-stage sampling, and a minimax allocation for group keys determined by expensive oracles. Algorithm 1 explicitly uses `SampleWithoutReplacement`. The local comparator uses known group keys and minimizes the largest plug-in rate variance. It allocates within each group in proportion to stratum size times estimated Bernoulli standard deviation. Independent second-stage draws with replacement yield a weighted Hoeffding interval at the declared budget. For stratum population size $N_h$ and second-stage sample size $m_h$, the count radius is $\sqrt{\frac12\log(2G/\delta)\sum_h N_h^2/m_h}$. Pilot calls count toward the budget. Repeated row draws count as calls. This adaptation replaces ABae's original estimators and bootstrap, so report it as an ABae-inspired comparator.

At 4,096 label calls on Adult, the mean of the largest per-group rate half-width was 12.62 percentage points for the fixed-budget comparator and 5.29 for StratQ. Both covered all groups in all 30 runs at each of the three tested budgets. The comparison varies the sampling design, confidence interval, and treatment of already observed labels together. [Results](results/adult_fixed_budget.json) · [Runner](experiments/adult_fixed_budget.py) · [Tests](tests/test_proxy_stratified.py)

## BlazeIt fixed-budget comparison

[BlazeIt, Section 6 and Appendix B.1](https://arxiv.org/html/1805.01046v2) uses proxy predictions as control variates and applies **always-valid** empirical Bernstein stopping to aggregate queries. The grouped adaptation here instead fits a fixed coefficient on up to 16 pilot labels per group, reveals small groups in full where the allocation permits, and draws independently with replacement from the remaining rows in other groups. It uses a fixed-look empirical Bernstein radius with an exact feature-derived range and equal per-group risk. Group budgets are predetermined for each requested total. Repeated draws count as calls, and distinct observed labels tighten the deterministic feasible interval. This comparator implements the control-variate aggregation and a single look; it must be called **BlazeIt-inspired fixed-look**, not BlazeIt's original anytime protocol. BlazeIt's EBGStop, proxy-only rewriting, and video optimizer are outside its scope.

Across 30 Adult replay seeds per budget, all nine group totals were covered in every run. The mean largest group rate half-width was 21.77, 11.92, and 3.49 percentage points at 2,048, 4,096, and 8,192 calls, respectively. StratQ's corresponding widths were 8.82, 5.29, and 2.73 points. The comparison changes sampling, allocation, proxy fitting, and interval construction together. [Raw results](results/adult_blazeit.json) · [Runner](experiments/adult_blazeit.py) · [Tests](tests/test_blazeit.py)

## Learned weighted sampling (LWS)

[Learning to Sample (PVLDB), §4.1 and Eq. (3)](https://arxiv.org/html/1906.09335#S4.SS1) uses probability-proportional-to-size draws without replacement, a positive floor on the proxy scores, and the ordered Des Raj estimate and sample variance. Independent exponential clocks implement the same sequential distribution in real arithmetic. The adaptation supplies the existing Adult proxy (profile-normalized), fixes the floor at 0.05, and assigns Senate quotas before sampling. We use the phase-two estimator only, not the training phase or learned stratified sampling. The local normal approximation splits nominal risk across groups and intersects the resulting interval with deterministic feasibility; census is exact and an empty intersection returns feasibility. These intervals are descriptive fixed-look comparators, not finite-sample certificates.

At 2,048 / 4,096 / 8,192 calls, the average maximum group rate half-widths are 7.69 / 4.45 / 1.65 percentage points, with joint coverage in 27 / 29 / 30 of 30 trials. [Runs](results/adult_lws.json) · [Operator](src/stratq/lws.py) · [Exhaustive small-sample checks](tests/test_lws.py)

## VerdictDB variational subsampling

[VerdictDB (SIGMOD 2018), §4 and Eq. (2) of the extended author manuscript](https://arxiv.org/html/1804.00770v3) constructs disjoint, variable-size subsamples and scales each aggregate's deviation by the square root of the realized subsample size. The adaptation operates on the same fixed-size uniform sample per group as Senate (matched row IDs, verified in a unit test). For a sample of size $m$, it uses $\lfloor\sqrt m\rfloor$ expected rows per bucket, $\lfloor m/\lfloor\sqrt m\rfloor\rfloor$ buckets, and an unused category for any remaining chance. It multiplies scaled mean deviations by the known group size divided by $\sqrt m$, inverts empirical equal-tailed quantiles around the group estimate, and clips to the feasible range. Empty buckets are omitted. A constant sample or fewer than two nonempty buckets returns feasibility; census is exact. There is no finite-sample or sequential guarantee, and we do not implement middleware, query rewriting, or join subsampling.

At 2,048 / 4,096 / 8,192 calls the mean maximum half-width is 6.27 / 3.88 / 1.65 percentage points, with joint coverage in 17 / 26 / 27 of 30 trials. [Runs](results/adult_verdict.json) · [Operator](src/stratq/verdict.py) · [Tests](tests/test_verdict.py)

## Hypergeometric comparison

[Waudby-Smith and Ramdas](https://arxiv.org/abs/2006.04347) construct a confidence sequence from a prior–posterior ratio. With a uniform prior on the unknown binary total, the predictive probability of observing $s$ positives in $t$ draws is $1/(t+1)$. The probability under a candidate total $K$ is hypergeometric. Retain $K$ when its likelihood is at least $\alpha_g/(t+1)$, then intersect over looks and with deterministic feasible bounds. Use equal risk allocation across groups. The floating-point implementation widens the log-likelihood cutoff by $10^{-8}$. The exact tests enumerate candidate totals and binary sample paths on small populations.

On Adult, the independent comparator used the same seed schedule, within-group permutations, batch sizes, and error budgets as the existing replay. Across 30 trials, it used 4,581.73 mean calls and covered all groups at the target width in each trial. StratQ without a proxy used 4,337.47 calls and the proxy-only variant used 4,338.53. The comparison measures label calls on this workload. [Results](results/adult_hypergeom.json) · [Runner](experiments/adult_hypergeom.py) · [Tests](tests/test_hypergeom.py)

## Sprint 2 modern conference operator adaptations

All five arms below are **operator-level adaptations**, not full system replicas. They run on the Sprint 1 safe-rounding engine or explicitly documented numerical safeguards. Source versions and contract differences are fixed here before any Sprint 2 result is quoted.

### Spertus-WoR

Source: [Spertus–Sridhar–Stark, arXiv:2409.06680v5](https://arxiv.org/html/2409.06680v5), §2.2 predictable interleaving, §2.3.1 WoR remaining-mean betting factor, §3.1 multiplicity-corrected simultaneous bounds. The local arm uses uniform within-group WoR, predictable max-slack cross-group routing, no ML control variate, and Bonferroni per-group risk. It reuses the safe `WoRControlVariateCS` with zero predictor. **Not implemented:** the union-of-intersections test sequence, hypothesis-dependent selections, or Proposition 2's weighted global LCB as a complete audit system. The comparison answers whether StratQ's residual control variate/refitting saves calls over raw stratified WoR inference.

### CELEUS-FixedProxy

Source: [Zhou et al., arXiv:2606.20820v3](https://arxiv.org/html/2606.20820v3), §3 Eq. (2) surrogate-completed IPW signal, Eq. (3)–(4) hedged-capital inversion, Appendix B Algorithm 2, Appendix C.1 Assumption 2 probability floor. The local arm fixes proxy scores before evaluation, implements the uniform WoR special case, and optionally uses `q_t(j)=β/m_t+(1−β)p_t(j)` with label-free acquisition scores. The target is finite-pool risk, so the population correction is disabled. Grid inversion is padded by one grid spacing as a conservative numerical guard. **Not implemented:** CELEUS's full LLM evaluation pipeline, population-risk lift, or oracle residual acquisition. The comparison isolates fixed-proxy e-process inference versus StratQ's predictable refitting and SQL decision contract.

### Calibrated-FixedProxy

Independent control, not a named paper replica. On the same pilot prefix it fits one Ridge linear predictor, freezes it, and then shares StratQ's random permutations, max-slack routing, batch sizes, risk budget and safe WoR CS. This blocks the objection that refitting only beats an uncalibrated raw score. The fixed theta is fitted once per group and never updated after the pilot.

### BARGAIN-Global

Source: [Zeighami et al., arXiv:2509.02896](https://arxiv.org/html/2509.02896v2), §4.1.2 Algorithm 3 and Appendix B.4.4 Lemma B.6 for global accuracy control; PACMMOD DOI [10.1145/3769776](https://doi.org/10.1145/3769776). BARGAIN is a statistical cascade method with global accuracy/precision/recall targets and adaptive sampling variants, not a database execution engine. The local arm replaces cascade threshold quality with a single global anytime count tolerance and reports proportional plug-in group outputs. **Not implemented:** LLM cascade routing, threshold search, or the paper's accuracy/precision/recall estimators. Its 0/30 all-group result measures only the per-group criterion on this surrogate; the JSON does not record a global interval, so it cannot test BARGAIN's own global guarantee.

### PilotDB-Asymp

Source: [Zhu et al., arXiv:2503.21087](https://arxiv.org/html/2503.21087v1), §2.4 simultaneous relative-error objective, §3.1 Theorem 3.1, Appendix B.1 CLT pilot asymptotics. Section 2.3 excludes aggregates used as GROUP BY keys (e.g., `GROUP BY COUNT(*)`), not ordinary COUNT outputs. The local arm uses a pilot→final normal interval with finite-population correction and Bonferroni group risk. An optional-stopping diagnostic reuses the fixed-look CLT radius at data-dependent looks; this is explicitly **not** a valid anytime certificate or PilotDB's intended protocol. **Not implemented:** PilotDB middleware, query rewriting, join/block sampling, or its exact-execution fallback. The comparison exposes small-sample and optional-stopping coverage failure modes; pooled tail coverage is 215/222=96.85%, not 94.7%.

Implementation: [modern_baselines.py](src/stratq/modern_baselines.py). Tests: [test_modern_baselines.py](tests/test_modern_baselines.py). Sprint 2 raw results are listed in [SPRINT2_REPORT.md](SPRINT2_REPORT.md).
