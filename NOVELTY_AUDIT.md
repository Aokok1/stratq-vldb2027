# StratQ novelty and citation-context audit

## Claim that survives the comparison

StratQ turns a predictable, proxy-adjusted finite-population observation into an operator for expensive grouped queries. The remaining-population correction allows proxy weights fitted on prior labels to change between batches. Conditional uniformity inside the selected group permits interleaving draws across groups. One joint coverage event governs grouped count intervals, threshold certificates, and ranking certificates, even at a data-dependent stop. None of proxy correction, empirical-Bernstein stopping, or allocation across groups is individually claimed to be new.

## Closest statistical antecedents

| Source and evidence | Overlap | Difference relevant to implementation |
|---|---|---|
| [Angelopoulos et al., Science 2023](https://doi.org/10.1126/science.adi6000); [extended manuscript, Appendices B and E.2](https://arxiv.org/html/2301.09633v4) | Uniform sample of labels on a fixed finite population; correct predicted means with a sampled residual. Appendix E.2 explicitly uses without-replacement betting in its nonasymptotic interval. | Fixed residual population in this construction. StratQ may refit the predictor after each batch and uses the mean of the current predictor over the remaining rows. A fixed-predictor confidence sequence is a valid comparator; finite-population prediction assistance alone is not a novelty claim. |
| [Csillag et al., ICML 2025, PMLR 267:11493–11514](https://proceedings.mlr.press/v267/csillag25a.html) | Anytime-valid, prediction-assisted estimation and testing via e-values. | Their e-value transformation and StratQ's remaining-population control variate are different constructions. This citation establishes existing anytime-valid prediction assistance, not a claim about everything the general framework can express. |
| [Kilian et al., NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/01830c92c6558179fa6d7fb1edff692c-Abstract-Conference.html) | Uniform-over-time asymptotic confidence sequences with prediction assistance and growing labeled and unlabeled sets. | StratQ's theorem is nonasymptotic in real arithmetic for fixed snapshot counts with known group keys, and it covers database decision certificates derived from those counts. |
| [Waudby-Smith and Ramdas, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html) | Without-replacement mean shifts and predictable empirical-Bernstein martingales. | StratQ does not claim a new concentration inequality; it specifies the current-observation correction and conditional mean for interleaved group queries. |

## Matched control suggested by the audit

For a fixed prediction vector f, define r_i = y_i−f_i and let F and F_seen be the sums of f over the full group and its revealed prefix. The conditional residual mean on the next draw is `(K − F − (S − F_seen))/m`. Adding `(F−F_seen)/m` yields `(K−S)/m`, exactly the StratQ conditional mean. Thus the word “remaining” by itself does not establish a difference from a correctly rebased fixed-predictor residual analysis. The [paired ablation](results/proxy_refit.json) holds the routing rule, per-group permutations, risk, batches, and coverage construction fixed while switching from the fixed normalized score to predictable refitting. The actual stopping time can differ. It measures the difference against the original score, not against an independently calibrated fixed coefficient. It is an ablation, not a PVLDB/SIGMOD baseline.

## Bibliographic boundaries

The amended bibliography uses the NeurIPS proceedings DOI for Kilian et al., the PMLR proceedings record for Csillag et al., and Science DOI metadata for Angelopoulos et al. The extended PPI preprint is the source used to check the appendices. For the Senate comparison the SIGMOD proceedings DOI is retained, rather than substituting the parallel SIGMOD Record DOI. The formal comparator tally is unchanged by these new citations. The search is targeted, not exhaustive; no first-of-kind claim follows. Full pre-submission retraction/version and citation-context checks remain open.
