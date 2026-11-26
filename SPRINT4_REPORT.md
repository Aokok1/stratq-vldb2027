# StratQ · Sprint 4 终稿红队、文献终验与 Artifact RC1 报告

日期：2026-11-26  
工作区：`E:\deepseek\submissions\vldb2027_stratq`  
范围：独立三角色 PC 模拟审稿、62 条文献联网终验、Must-Fix 闭环、终稿硬门禁、公开 Artifact 候选包。未执行 git push、公开上传或 CMT 在线提交。

## 1. 总结

Sprint 4 已完成两轮红队与全部可本地闭环的 Must-Fix。原稿的主要均值可信，但内存、延迟、基线归因、Yelp 机制解释和若干文献能力标记会实质误导审稿人，现已按原始 JSON/官方原文修正。

关键变化：

- **Yelp COUNT 负结果保留，但原机制解释撤回。** 202 次 gate 决策中 112 次选择 proxy、75 次无成本改进回退、15 次历史不足；平均 LOO 方差下降 12.9%，但最大 prediction span 仅 $8.9\times10^{-16}$，无 $R>1.01$。范围惩罚解释不被支持。
- **10M 内存口径纠正。** query 子进程总墙钟 187.85 s = profile rebuild 103.50 s + query-only 37.86 s。+21.69 MB 是一次瞬时 pre-teardown RSS 差；query traced-allocation peak 为 85.6 MB；最终 post-teardown RSS 差为 −0.68 MB。摘要不再把它写成 retained/query-only memory。
- **GPU 延迟边界公开。** 3.61×–4.78× 是代理分数预计算、模型/OS cache 预热后的 ratio-of-medians query-level speedup。Yelp 加入 proxy scan 后为 3.05×/3.09×，首次再加入 training 为 2.41× fresh。
- **对照归因纠正。** BARGAIN-Global 是全局 count surrogate，0/30 只表示逐组 plug-in 不覆盖，不是 BARGAIN 原全局保证失败；PilotDB optional-stop 是 fixed-look CLT diagnostic；BlazeIt 本地 arm 是 inspired fixed-look；决策基线是 post-hoc COUNT-derived，不是等预算专用 routing。
- **统计与设计口径补齐。** 3-seed COUNT 补 sample SD/range；Zipf 明确 15 configurations × 2 seeds；960 decision trials 写全设计因子；SMS 补 legacy 行并承认 safe_loo 不优于 legacy；CPU timing 用对齐协议重跑。
- **数学/实现一致性修复。** 证明包下端点改为减 $\ell^+$；正文重启规则与 Theorem 2/代码统一；Decimal log 增加相邻十进制端点；归一化标签传播完整 outward interval；Higham 总量上界改为 $\gamma_NN$。
- **文献终验闭环。** 62/62 均被引用；补 NeurIPS 页码 11664–11673、JMLR 27(19):1–90、ICLR 2026 Poster；8 篇 2026 预印本保持 preprint；Spertus 明确 submitted to AOAS、非录用；官方/API/定向筛查未见确认撤稿/EoC/重大勘误信号。

## 2. 交付文件

| 文件 | 内容 |
|---|---|
| [`VLDB_PC_REVIEW_SPRINT4.md`](VLDB_PC_REVIEW_SPRINT4.md) | Reviewer 1–3 + Meta-Reviewer Round 1、Must-Fix/Polish、Round 2 逐条关闭 |
| [`BIBLIOGRAPHY_AUDIT.md`](BIBLIOGRAPHY_AUDIT.md) | 62 条在线状态、撤稿筛查、引用上下文与官方来源 |
| [`PROOF_PACKAGE.md`](PROOF_PACKAGE.md) / [`PROOF_AUDIT.md`](PROOF_AUDIT.md) | 修复后的证明、数值 enclosure 与回归记录 |
| [`stratq.pdf`](4_LaTeX_Source/vldb_template/stratq.pdf) | 12 页正文 + 参考文献，14 页总计 |
| [`PAPER_CLAIM_AUDIT_SPRINT3.json`](4_LaTeX_Source/vldb_template/PAPER_CLAIM_AUDIT_SPRINT3.json) | 扩展后的 88/88 Claim-to-JSON/结构门禁 |
| [`sprint4_summary_statistics.json`](results/sprint4_summary_statistics.json) | 从原始 JSON 重算 SD/range/分母/计时与内存定义 |
| [`sprint4_yelp_gate_diagnostic.json`](results/sprint4_yelp_gate_diagnostic.json) | Yelp Safe-LOO trace 与机制检验 |
| [`sprint4_cpu_latency_aligned.json`](results/sprint4_cpu_latency_aligned.json) | 对齐 CPU exact/fresh/reused 复测 |
| [`sprint4_code_change_validation.json`](results/sprint4_code_change_validation.json) | 数值修复前后 Adult calls/decisions/端点比较 |
| [`ARTIFACT.md`](ARTIFACT.md) | 离线审计、固定 revision/SHA-256 获取、复现边界 |
| `stratq-artifact-vldb2027-rc1.zip` | 干净自包含 RC1 候选包；独立解压验证结果见 §4 |

## 3. 硬门禁结果

| 门禁 | 结果 |
|---|---|
| 源码全套 `pytest` | **52 passed, 3 subtests passed, 0 failed**（203.15 s） |
| LaTeX | `latexmk -g -pdf -interaction=nonstopmode -halt-on-error stratq.tex` 成功 |
| 正文页数 | **12 页整**；REFERENCES 从第 13 页顶部开始；PDF 共 14 页 |
| LaTeX error | **0** |
| undefined citation/reference | **0**（另有 1 个 font-shape warning，不是引用/参考文献未定义） |
| Overfull hbox ≥5 pt | **0**（保留 4.15309 pt hbox 与 1.17 pt vbox，均低于门槛） |
| Claim-to-JSON/结构审计 | **88/88 passed, 0 mismatch** |
| CMT Title/Abstract/SAR | **逐字一致**；作者占位与空 availability URL 保留 |
| 文献结构 | **62 active / 62 cited / 0 undefined / 0 uncited** |
| 视觉检查 | 14 页逐页/拼图检查；无遮挡、裁切或图号错位；作者块/URL 填入后仍须复核 |
| 输入获取链 | Adult/SMS/Fashion/Yelp/CIFAR/ResNet34/DistilBERT 全部固定 SHA-256 本地验证通过 |

## 4. Artifact RC1 独立解压验证

已在全新系统临时目录解压 RC1，并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，依次执行：

| 命令 | 结果 |
|---|---|
| `python artifact_tools.py verify` | 134 个 payload 哈希全部通过 |
| `python -m pytest -q` | 52 passed, 3 subtests passed |
| `python experiments/audit_bibliography_sprint4.py` | 通过 |
| `python experiments/audit_submission_sprint4.py` | Title/Abstract/SAR/占位符全部通过 |
| `python experiments/audit_paper_claims_sprint3.py` | 88/88 通过 |

ZIP CRC、统一前缀、无 `data/`、`models/`、`__pycache__`、`.pytest_cache`、权重/Parquet/旧 ZIP 负载检查均通过。临时解压目录已删除。机器可读记录：[`artifact_independent_verification_sprint4.json`](results/artifact_independent_verification_sprint4.json)。该记录保留在工作区，不嵌入 ZIP，以避免验证哈希循环变化。

## 5. 边界与未完成项

- 未重跑数小时 GPU 大实验；保留原始冻结 JSON，新增 JSON 均为派生统计或分钟级轻量诊断。
- 当前代码的归一化 enclosure 修复会向外放宽端点；Adult 回放 calls/decisions 不变，最大端点变化约 $2.9\times10^{-11}$ count units。冻结 JSON 保留原代码哈希，不伪造“当前代码全量重跑”。
- 3-seed/2-seed 结果是描述性证据；不是总体覆盖率或显著性证明。
- Artifact 是候选包，不是已公开仓库、Zenodo DOI 或 PVLDB reproducibility badge。
- 作者身份、CoI/Light Load Reviewer、代码许可与公开 URL、最终 CMT 操作仍未执行，见 [`SUBMISSION_PREFLIGHT.md`](SUBMISSION_PREFLIGHT.md) 的四项人工开关。
