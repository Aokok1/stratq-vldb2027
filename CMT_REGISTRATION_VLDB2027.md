# VLDB 2027 · CMT registration card

**Track:** PVLDB Vol. 20 Regular Research, January 2027 cycle. [CMT](https://cmt3.research.microsoft.com/PVLDBv20_2027/) · [Important dates](https://vldb.org/2027/important-dates.html) · [Submission guidelines](https://vldb.org/2027/submission-guidelines.html)

Copy the title, abstract, and SAR below into their respective fields after confirming the author-controlled items. Nothing has been registered or submitted.

| Milestone | Pacific Time | Beijing time |
|---|---|---|
| Mandatory abstract | 2026-12-25 17:00 PST | 2026-12-26 09:00 CST |
| Paper submission | 2027-01-01 17:00 PST | 2027-01-02 09:00 CST |

Initial notification is scheduled for February 15, 2027. Regular Research permits at most 12 pages excluding references (appendices and acknowledgments count toward the limit). Review is single-blind: real author names and affiliations must appear on page one.

## Title

StratQ: Anytime-Valid Grouped Query Contracts for Expensive Predicates

## Abstract

Analytical SQL queries over text, images, and structured records increasingly evaluate expensive neural or language-model predicates inside GROUP BY count, HAVING, and top-k operators. Evaluating the reference model on every row dominates query latency, yet existing proxy-assisted and approximate query engines typically protect only an overall aggregate, assume fixed sample sizes or large-sample asymptotics, or omit decision certification under early stopping. We present StratQ, a query execution engine and statistical contract for fixed-snapshot grouped queries with expensive binary predicates. Under uniform sampling without replacement within each adaptively selected group, StratQ couples remaining-population centering with a predictable, analytically cross-validated Ridge control variate that reverts to a zero-range baseline unless its leave-one-out cost surrogate improves by at least 5%. All group intervals are computed via directed binary64 outward rounding and share a single (1-delta) simultaneous coverage event across all groups and data-dependent reveal times, directly certifying COUNT tolerances, HAVING thresholds, and top-k membership while permitting explicit abstention under budget caps. Implemented over DuckDB and Apache Arrow and evaluated across six workload families (including live GPU DistilBERT-SST2 and ResNet34-224 inference and 10M-row Parquet snapshots) and eleven database/modern operator adaptations plus an independent grouped hypergeometric CS, StratQ delivers 3.61x--4.78x ratio-of-medians query-level GPU speedups after proxy scoring, certifies top-2 membership on Yelp with 3.0% of rows (6.5x fewer calls than raw stratified WoR), observes 30/30 all-group coverage over 15 skewed Zipf configurations (two seeds each) versus 0/30 for a global-count surrogate's group plugs and 16/30 for an optional-stopping CLT diagnostic, and records an 85.6 MB query-stage traced-allocation peak at 10M rows.

## Self-Assessment of Relevance (SAR)

StratQ does not claim a new concentration inequality, nor does it claim that a proxy always accelerates every query. Instead, its contribution is an end-to-end, database-facing grouped query contract and execution engine that unifies predictable batch-refitted residual control variates, uniform within-group WoR sampling, cross-group adaptive routing, and directed binary64 outward rounding under a single filtration and (1-delta) joint coverage event.

We evaluate StratQ across six workload families and eleven database/modern operator adaptations plus an independent grouped hypergeometric CS to answer five questions: (Q1) Does StratQ achieve real end-to-end latency speedups and reliable decision certification on expensive GPU neural predicates? (Q2) How does StratQ compare against modern 2025--2026 statistical and database comparators (Spertus-WoR, CELEUS-FixedProxy, Calibrated-FixedProxy, BARGAIN-Global, PilotDB-Asymp)? (Q3) Does Safe-LOO range--variance gating deliver strong savings on accurate proxies without degrading under noisy or weak proxies? (Q4) How do StratQ's guarantees and cost behave across classical grouped AQP baselines, decision boundaries, and high group counts under Zipf skew? (Q5) How do DuckDB+Arrow profiling, query reuse, and isolated memory footprint scale to 10 million rows?

StratQ reframes proxy-assisted analytical querying around an explicit, anytime-valid grouped SQL execution contract. By combining uniform within-group WoR sampling, predictable Safe-LOO residual control variates, cross-group adaptive routing, and directed binary64 outward rounding under a single (1-delta) simultaneous coverage event, StratQ unifies grouped COUNT estimation, HAVING classification, and TOP-k membership certification with explicit abstention.

## Suggested topics

Approximate query processing; query execution and optimization; aggregation; data management for machine learning.

## Author-controlled details (required before submission)

- Replace `Author information pending` in `stratq.tex` with real author names, affiliations, and emails; match CMT exactly.
- Register all authors in CMT and declare individual and institutional Conflicts of Interest (CoI).
- Nominate at least one author as Light Load Reviewer.
- Publish the reproducibility artifact to a public archival repository (GitHub/Zenodo), paste its URL into `\vldbavailabilityurl{}` and CMT, and verify signed-out access.
- Verify submission caps (2/month, 12/Vol. 20), originality, and any concurrent-submission disclosures.
