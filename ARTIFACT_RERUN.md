# Same-host clean-extraction rerun

The full round-19 experiment suite ran from a fresh Windows x64 extraction and virtual environment. This checks that another extraction can regenerate the evidence; it is not independent, cross-machine reproduction or an artifact badge.

- Input package: `stratq-review-r19.zip`, SHA-256 `5363b5984c7c9cc5d898f336390435ee71467a83158db450e29f6d465a169194`.
- Verified 74 payload hashes and six freshly downloaded data inputs. Pinned dependencies installed and passed `pip check`. All 17 unit tests and 77 frozen claim checks passed before rerunning.
- Ran Adult and its three venue-derived comparators, grouped hypergeometric comparator and routing ablation; SMS and Fashion-MNIST live experiments; proxy refitting; latency; decision stress tests; and 100K/1M/10M-row replay. All commands, result audits, and figure regeneration exited successfully. The extracted PDF rebuilt to five US Letter pages with fonts embedded.
- [Result comparison](experiments/compare_rerun.py) reported zero non-timing differences for 14 result sets after checking source hashes and comparing floats with tolerance. JSON integer versus floating-point representation alone was not treated as a difference (e.g., `358` versus `358.00000000000006` at a feasible endpoint). A negative control that changes a call count by one was detected.

## Timings did not reproduce exactly

| Measurement | Frozen | Fresh extraction |
|---|---:|---:|
| SMS fastest tested exact median | 872 ms, one batch | 500 ms, one batch |
| SMS fresh-profile proxy median | 1,425 ms | 882 ms |
| Fashion-MNIST fastest tested exact median | 1,961 ms, batch 512 | 1,418 ms, batch 1,024 |
| Fashion-MNIST fresh-profile proxy median | 646 ms | 498 ms |
| 10M-row profile | 32.64 s | 25.57 s |
| 10M-row warm replay median: no proxy / proxy | 1,521 / 1,496 ms | 1,044 / 1,075 ms |

The exact-versus-fresh-proxy ordering held for both live workloads. The best exact batch size on Fashion-MNIST changed and the direction of the small proxy replay timing difference at 10M rows reversed. The 10M-row reference-call reduction (33,280 to 29,610.67 on average) was unchanged. The manuscript retains the frozen measurements, not a mixture of selected cells from the two runs.

## Extension to the current draft

After the original rerun, the Online Aggregation and capped Adult routing studies were run in the same isolated environment with only their new code and tests added. All 20 tests passed and neither new study differed from its frozen record. The LWS and VerdictDB studies were then reproduced in the same pinned environment. All 24 tests passed and the expanded [comparator](experiments/compare_rerun.py) returned zero non-timing differences across all 18 result sets. This is not a second full rerun from the newer ZIP.
