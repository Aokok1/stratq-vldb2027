# Bibliography verification record — Sprint 4

核验日期：**2026-11-26**  
对象：[`stratq_refs.bib`](4_LaTeX_Source/vldb_template/stratq_refs.bib) 的全部 **62 个 active entry**，以及 `stratq_problem.tex`、`stratq_evaluation.tex`、`stratq_discussion.tex` 中全部引用上下文。

## 1. 总结

- 62/62 active entry 均被正文引用；三个指定 TeX 文件共 111 次引用、62 个不同 key；无 undefined key，也无未引用 entry。
- 当前状态：**53 条正式发表著作/论文、8 条预印本、1 条软件项目记录**。
- 本轮补齐/升级 3 条元数据：
  - `tao2019threshold`：NeurIPS 2019 正式页码 **11664–11673**；
  - `jourdan2026apgai`：正式 **JMLR 27(19):1–90, 2026**；
  - `sfyraki2026active`：正式 **ICLR 2026 Poster**，OpenReview `Iw0tMeLed8`。
- `spertus2026sequential` 最新为 arXiv v5（2026-08-25），comments 仍为 **submitted to Annals of Applied Statistics**；未发现正式录用/发表证据，不能写 forthcoming。
- 其余 7 篇 2026 预印本截至当前未发现正式发表更新。
- 撤稿/EoC/重大勘误：**官方记录、API 与定向网页筛查中未发现确认信号**。这不是“绝对无撤稿”的证明，限制见 §3。
- 上下文审计发现 PilotDB、BlazeIt、ABae、BARGAIN、Spertus、Hait、CELEUS 等若干过度授信/错误轴定义，已在 Sprint 4 Must-Fix 中闭环或要求闭环。

## 2. 四项开放项的正式证据

### `tao2019threshold`

[NeurIPS 官方 Metadata.json](https://proceedings.neurips.cc/paper_files/paper/2019/file/9a0684d9dad4967ddd09594511de2c52-Metadata.json) 给出 `page_first=11664`、`page_last=11673`。官方 BibTeX 的 pages 字段为空，但 metadata 是正式 proceedings 记录。已补 `pages={11664--11673}`。

注意：常见检索摘要中的 11973–11982 对应另一篇 bandit 论文，不是本文。

### `jourdan2026apgai`

[JMLR 官方页面](https://jmlr.org/papers/v27/24-0680.html) 与[官方 BibTeX](https://jmlr.org/papers/v27/24-0680.bib) 一致：Jourdan、Delahaye-Duriez、Réda，*An Anytime Algorithm for Good Arm Identification*，JMLR **27(19):1–90, 2026**。已改为 `@article`，删除过时的 arXiv “to be published” note。

### `sfyraki2026active`

OpenReview 官方 forum [`Iw0tMeLed8`](https://openreview.net/forum?id=Iw0tMeLed8) 的记录为 **ICLR 2026 Poster**，venueid `ICLR.cc/2026/Conference`，已有 camera-ready revision。已改为 `@inproceedings`，booktitle 为 *The Fourteenth International Conference on Learning Representations*；不虚构页码或 DOI。OpenReview 页面可能显示浏览器验证，状态以官方 API 记录为准。

### `spertus2026sequential`

[arXiv 记录](https://arxiv.org/abs/2409.06680) 最新为 **v5，2026-08-25**；comments 仍为 “submitted to Annals of Applied Statistics”，无 `journal_ref`、正式 DOI、卷期页码。Crossref/OpenAlex 对应记录仍为 preprint。BibTeX note 已明确“submitted”，未改写为录用。

### 其他 2026 预印本

| key | 当前版本 | 正式发表更新 |
|---|---:|---|
| `zhou2026celeus` | v3，2026-11-07 | 未发现 |
| `hait2026target` | v2，2026-10-08 | 未发现 |
| `khosravi2026rankcs` | v1，2026-09-26 | 未发现 |
| `wu2026faq` | v4，2026-10-29 | 未发现 |
| `shekhar2026rates` | v1，2026-03-15 | 未发现 |
| `xu2026fppi` | v1，2026-02-11 | 未发现 |
| `dou2026conformalcascade` | v3，2026-07-31 | 未发现 |

以上均无 arXiv `journal_ref` 或正式出版 DOI；OpenAlex 仍标为 preprint。

## 3. 撤稿、关切声明与勘误筛查

本轮采用：

1. 43 个 DOI 的 Crossref 记录：检查 `update-to`、`relation`、类型与日期；
2. OpenAlex 对应记录：检查 `is_retracted`；
3. OpenAlex 引用文献标题筛查：共扫描 5,366 篇，单篇高被引论文最多 400 篇；
4. 官方 proceedings、JMLR、OpenReview、arXiv、出版社 DOI 页与针对性搜索。

结果：

- Crossref 43/43 DOI 均无 `update-to`；
- `relation` 仅见重复 proceedings DOI、supplement、book chapter 等正常关系；
- OpenAlex 未见对应记录 `is_retracted=true`；
- 标题筛查 6 个词面命中经人工检查均为无关的 “concern/correction/error correction” 用法；
- 官方页面与定向搜索未发现确认的 retraction/EoC/重大勘误。

**限制：** API 链接和标记不保证完整；出版社未链接的更正、编辑部信函、非 DOI 会议记录及尚未入库的最新声明可能漏检。因此结论限定为“截至 2026-11-26 的筛查未见确认信号”。

## 4. 完整 62 条核验表

记号：**NSS** = 上述筛查未见确认的撤稿/EoC/重大勘误信号；软件记录不适用 NSS。DOI 链接是官方解析入口，无 DOI 者使用官方 proceedings/OpenReview/JMLR/CIDR/arXiv/软件官网。

| key | 状态 | 官方 URL | 动作/上下文备注 |
|---|---|---|---|
| `kang2021abae` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3476249.3476285) | 元数据无需变更；NSS；Table 1 Fixed WoR 应修正，见 C2 |
| `shadow2023` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3625054.3625059) | 无；NSS |
| `sample2016` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/2882903.2915249) | 无；NSS |
| `wor2020` | 正式·NeurIPS | [官方页面](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html) | 33:20204–20214 正确；NSS |
| `kang2019blazeit` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3372716.3372725) | 元数据无需变更；NSS；原文是 always-valid EBS，见 C2 |
| `acharya2000congress` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/342009.335450) | proceedings DOI 正确；NSS；Senate 为本地 adaptation |
| `angelopoulos2023ppi` | 正式·Science | [DOI](https://doi.org/10.1126/science.adi6000) | 无；NSS；扩展附录确有 WoR betting |
| `csillag2025evalues` | 正式·ICML | [PMLR](https://proceedings.mlr.press/v267/csillag25a.html) | 267:11493–11514 正确；NSS |
| `kilian2025anytime` | 正式·NeurIPS | [DOI](https://doi.org/10.52202/085713-0033) | 38:1035–1053 正确；NSS |
| `hellerstein1997online` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/253260.253291) | 171–182 正确；NSS；基线是 scan adaptation |
| `walenz2019lws` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3368289.3368302) | 13(3):390–402 正确；NSS |
| `park2018verdict` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/3183713.3196905) | 1461–1476 正确；NSS |
| `haas1999ripple` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/304182.304208) | 287–298 正确；NSS |
| `li2016wander` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/2882903.2915235) | 615–629 正确；NSS |
| `sheoran2023deepola` | 正式·PACMMOD | [DOI](https://doi.org/10.1145/3589269) | 1(2):1–28 正确；NSS |
| `russo2023inquest` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3611479.3611496) | 16(11):2897–2910 正确；NSS；O(1/B) 有 stationary-stream 条件 |
| `arai2009anytimetopk` | 正式·VLDB Journal | [DOI](https://doi.org/10.1007/s00778-008-0127-9) | 18(2):407–427 正确；NSS |
| `jun2016atlucb` | 正式·ICML | [PMLR](https://proceedings.mlr.press/v48/jun16.html) | 48:974–982 正确；NSS |
| `tao2019threshold` | 正式·NeurIPS | [Metadata](https://proceedings.neurips.cc/paper_files/paper/2019/file/9a0684d9dad4967ddd09594511de2c52-Metadata.json) | **已补 11664–11673**；NSS |
| `kang2020supg` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3407790.3407804) | 元数据无需变更；NSS；保证应注明 asymptotic/CLT |
| `jourdan2026apgai` | 正式·JMLR | [JMLR](https://jmlr.org/papers/v27/24-0680.html) | **已改 @article，27(19):1–90**；NSS |
| `khosravi2026rankcs` | 预印本 v1 | [arXiv](https://arxiv.org/abs/2609.32211) | 无正式更新；NSS；top-k/early retirement 陈述成立 |
| `raasveldt2019duckdb` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/3299869.3320212) | 1981–1984 正确；NSS |
| `apachearrow` | 软件 | [Arrow 规范](https://arrow.apache.org/docs/format/Columnar.html) | 软件记录；撤稿筛查不适用 |
| `patel2025lotus` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3749646.3749685) | 18(11):4171–4184 正确；NSS；能力切割见 C2 |
| `arnold2026smcs` | 正式·JRSSB | [DOI](https://doi.org/10.1093/jrsssb/qkag066) | 88(5):1662–1681 正确；NSS |
| `gu2026serpant` | 正式·ICML | [PMLR](https://proceedings.mlr.press/v306/gu26l.html) | 306:37229–37260 正确；NSS |
| `zhou2026celeus` | 预印本 v3 | [arXiv](https://arxiv.org/abs/2606.20820) | 2026-11-07；无正式更新；NSS；见 C6 |
| `spertus2026sequential` | 预印本 v5 | [arXiv](https://arxiv.org/abs/2409.06680) | **note 已明确 submitted to AOAS，非录用**；NSS；见 C4 |
| `hait2026target` | 预印本 v2 | [arXiv](https://arxiv.org/abs/2609.37873) | 2026-10-08；无正式更新；NSS；见 C5 |
| `waudby2024betting` | 正式·JRSSB | [DOI](https://doi.org/10.1093/jrsssb/qkad009) | 86(1):1–27 正确；online 2023、print 2024；NSS |
| `wu2026faq` | 预印本 v4 | [arXiv](https://arxiv.org/abs/2601.20251) | 2026-10-29；无正式更新；NSS；主定理要求 with-replacement |
| `zrnic2024active` | 正式·ICML | [PMLR](https://proceedings.mlr.press/v235/zrnic24a.html) | 235:62993–63010 正确；NSS |
| `fisch2024stratppi` | 正式·NeurIPS | [DOI](https://doi.org/10.52202/079017-3541) | 37:111489–111514 正确；NSS |
| `turner2023stratified` | 正式·AISTATS | [PMLR](https://proceedings.mlr.press/v206/turner23a.html) | 206:4880–4893 正确；NSS |
| `sfyraki2026active` | 正式·ICLR | [OpenReview](https://openreview.net/forum?id=Iw0tMeLed8) | **已改 @inproceedings，ICLR 2026 Poster**；NSS |
| `zrnic2024crossppi` | 正式·PNAS | [DOI](https://doi.org/10.1073/pnas.2322083121) | 121(15):e2322083121 正确；NSS |
| `angelopoulos2026ppipp` | 正式·AOAS | [DOI](https://doi.org/10.1214/26-AOAS2215) | 20(3):2235–2252 正确；NSS |
| `shekhar2026rates` | 预印本 v1 | [arXiv](https://arxiv.org/abs/2603.14423) | 2026-03-15；无正式更新；NSS；non-sequential 判断正确 |
| `waudby2021rilacs` | 正式·E-Vote-ID | [DOI](https://doi.org/10.1007/978-3-030-86942-7_9) | LNCS 12900:124–139 正确；NSS |
| `orabona2024portfolio` | 正式·IEEE TIT | [DOI](https://doi.org/10.1109/TIT.2023.3330187) | 70(1):436–455 正确；online 2023、print 2024；NSS |
| `howard2021cs` | 正式·AOS | [DOI](https://doi.org/10.1214/20-AOS1991) | 49(2):1055–1080 正确；NSS |
| `xu2026fppi` | 预印本 v1 | [arXiv](https://arxiv.org/abs/2602.10464) | 2026-02-11；无正式更新；NSS；主要效率结论为 asymptotic |
| `ramdas2023savi` | 正式·Statistical Science | [DOI](https://doi.org/10.1214/23-STS894) | 38(4):576–601 正确；NSS |
| `breidt2017modelassisted` | 正式·Statistical Science | [DOI](https://doi.org/10.1214/16-STS589) | 32(2):190–205 正确；NSS |
| `sarndal1992model` | 正式·专著 | [DOI](https://doi.org/10.1007/978-1-4612-4378-6) | 无；NSS |
| `agarwal2013blinkdb` | 正式·EuroSys | [DOI](https://doi.org/10.1145/2465351.2465355) | 29–42 正确；NSS |
| `kandula2016quickr` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/2882903.2882940) | 631–646 正确；NSS |
| `goiri2015approxhadoop` | 正式·ASPLOS | [DOI](https://doi.org/10.1145/2694344.2694351) | 383–397 正确；NSS |
| `wen2018approxiot` | 正式·ICDCS | [DOI](https://doi.org/10.1109/ICDCS.2018.00048) | 411–421 正确；NSS |
| `peng2018aqpplus` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/2882903.2882940) | 1477–1492 正确；NSS |
| `zhu2025pilotdb` | 正式·PACMMOD | [DOI](https://doi.org/10.1145/3725335) | 3(3):1–28 正确；NSS；GROUP BY 限制见 C1 |
| `kang2017noscope` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3137628.3137664) | 10(11):1586–1597 正确；NSS |
| `lu2018pp` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/3183713.3183751) | 1493–1508 正确；NSS |
| `yang2022core` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3547305.3547310) | 15(10):2032–2044 正确；NSS |
| `kang2022tasti` | 正式·SIGMOD | [DOI](https://doi.org/10.1145/3514221.3517897) | 1934–1947 正确；NSS |
| `dai2024uqe` | 正式·NeurIPS | [DOI](https://doi.org/10.52202/079017-0938) | 37:29807–29838 正确；NSS；Crossref 作者缺失，以官方/arXiv 为准 |
| `liu2025palimpzest` | 正式·CIDR | [CIDR 官方列表](https://cidrdb.org/cidr2025/papers.html) | accepted paper；NSS |
| `shankar2025docetl` | 正式·PVLDB | [DOI](https://doi.org/10.14778/3746405.3746426) | 18(9):3035–3048 正确；NSS |
| `jo2024thalamusdb` | 正式·PACMMOD | [DOI](https://doi.org/10.1145/3654989) | 2(3):1–26 正确；NSS |
| `zeighami2026bargain` | 正式·PACMMOD | [DOI](https://doi.org/10.1145/3769776) | 正式年份 **2025**，3(6):1–26；key 保留；NSS；见 C3 |
| `dou2026conformalcascade` | 预印本 v3 | [arXiv](https://arxiv.org/abs/2607.25018) | 2026-07-31；无正式更新；NSS；边际保证有条件，见 C8 |

## 5. 引用上下文核验

三个 TeX 文件中全部含引用段落均已检查。常规归属正确的段落包括：DB AQP 谱系（Congressional、Sample+Seek、Quickr、ShadowAQP、BlinkDB、AQP++、VerdictDB、LWS、ApproxHadoop/IoT）；昂贵谓词谱系（NoScope、Probabilistic Predicates、CORE、TASTI、SUPG、BlazeIt、InQuest）；在线聚合与 join；bandit/ranking；WoR/SAVI/portfolio/RiLACS；PPI/Active/FAQ/FPPI；DuckDB/Arrow。

关键上下文问题与处置如下。

### C1. PilotDB 的 GROUP BY 排除项

[PilotDB §2.3](https://arxiv.org/html/2503.21087) 原文排除的是 **aggregates in Group By clauses (e.g., GROUP BY COUNT(*))**，即 aggregate 本身作为分组键。它不笼统排除 `SELECT category, COUNT(*) ... GROUP BY category`。其保证是预先指定 sampling plan 下的同时相对误差，依赖 CLT/大样本，不是 anytime，也不直接提供 StratQ 式 HAVING/精确 TOP-k 成员认证。

**处置：** 删除“excludes queries whose aggregate appears in a GROUP BY COUNT(*) decision pattern”，改为准确限定；PilotDB-Asymp 的 optional-stop 结果明确标为 fixed-look CLT diagnostic。

### C2. BlazeIt、ABae、LOTUS

- BlazeIt 原文明确使用 **always-valid Empirical Bernstein stopping**。正文只能把本地固定预算 arm 称为 “BlazeIt-inspired fixed-look”，不能称原论文 fixed-look。
- ABae Algorithm 1 明确 `SampleWithoutReplacement`，故 Table 1 `Fixed WoR=×` 错误；其为两阶段固定预算/bootstrap CI，不是 anytime。ABae-GroupBy 的 minimax allocation 不等同持续 cross-group routing。
- LOTUS 的 fixed-sample CLT/multiplicity 描述正确；`sem_group_by` 发现未知语义类别，`sem_topk` 排序语义行，不是已知 key 的组计数 CI/top-k。LOTUS 未在本地实测。

### C3. BARGAIN

BARGAIN 原文保证整个输出的 accuracy/precision/recall，不是逐组保证，且有 adaptive sampling 与 without-replacement 变体。本地 `BARGAIN-Global` 替换为全局 count tolerance + proportional plug-in，不是 cascade replica。0/3、0/30 只说明该本地 global-count surrogate 不满足逐组准则，**不能说明 BARGAIN 原保证失败**。Table 1 的 DB Engine=✓ 无依据，Routing 需按“跨组路由”定义处理。

### C4. Spertus

Spertus 支持有限总体、层内 WoR、predictable cross-stratum selection、remaining-mean、同时 LCB 与 union-of-intersections TSM；但 UI-TS 不共享单一 global TSM，主要输出是假设检验/LCB，不是全组双侧 COUNT 区间或 SQL HAVING/TOP-k。`Spertus-WoR` 本地基线未实现 UI-TS。

### C5. Hait

Hait 使用 i.i.d./paired within-stratum 假设、预设 target mixture/target set、stratum plug-in augmentation 与 best-system identification；不直接提供固定有限总体的任意 known-key COUNT/HAVING/精确 grouped TOP-k。Table 1 的 Refit CV=pmark 与 HAVING/TOP-k=✓ 过宽。

### C6. CELEUS

CELEUS 支持 fixed-pool WoR、fixed surrogate completion、IPW residual、hedged-capital e-process 与 uncertainty-guided acquisition，但针对单一风险指标，不是多组 SQL 引擎。Table 1 Routing=× 若指跨组路由才成立；HAVING/TOP-k=pmark 应降为 ×。本地跨组实现与弱 proxy 覆盖结果属于 adaptation。

### C7. PPI / Active / FAQ / FPPI

PPI 扩展附录确有 WoR betting；Active Statistical Inference 允许 predictable refitting；FAQ 主定理要求 with-replacement 与 martingale CLT；FPPI 支持“噪声预测需选择性使用”的定性结论，但主要结果是 asymptotic efficiency，不是 Safe-LOO anytime 证明。

### C8. Rank CS / SERPANT / SMCS / Conformal Cascade

Rank CS 支持 optional retirement、rank sets、top-k membership，但允许并列导致超过 k 个模型，不等同 StratQ 的精确 k-element set + census tie-break。SERPANT/SMCS 归属正确。Conformal Cascade 通用保证为 \(1-K\alpha\)，selection-preservation 条件下才收紧到 \(1-\alpha\)，且非 sequential anytime。

### C9. Table 1 比较轴

原轴过于笼统。Sprint 4 改为：固定有限总体 WoR、非渐近 anytime、可预测 batch-refit residual CV、跨组自适应路由、逐组计数区间、SQL 决策认证、DB 执行集成，并据 C1–C8 修正单元格。

### C10. 实验基线归因

五个 modern arms 都是 operator-level adaptations：Spertus-WoR 未实现 UI-TS；CELEUS 未实现完整 LLM pipeline/population lift；BARGAIN-Global 替换原目标；PilotDB-Asymp 未实现 middleware/query rewriting。图注、表注和结果句必须持续使用 adapted/global-count surrogate/fixed-look diagnostic，不能把本地失败率归因于原系统。

### C11. 论证措辞

- 全局 tolerance 不**必然**导致 rare groups 失败，只是不蕴含逐组保证；0/30 是本地 surrogate 实验。
- CLT 重复 look **不保证**名义覆盖且可能反保守，不应写成数学上必然违反。
- “首个 anytime top-k”不成立；StratQ 的新颖性严格限定为固定快照 SQL 组计数、WoR、refit CV、joint event、abstention 与 DB 执行组合。

## 6. 提交前仍须人工关注的项

1. Camera-ready 前重查 8 篇预印本，尤其 Spertus、CELEUS、Hait。
2. 若 venues 要求出版社页面级完整撤稿核验，当前 API/标题筛查仍不能替代 53 条逐一人工出版社复核。
3. 作者须确认最终引用上下文与 Table 1 修订，不得把本地 adaptation 改名成原系统。
4. `CITE_CHECK_STATIC.json` 是 2026-11-24 的 10 条 parse-only 历史记录，**不是**本轮 62 条在线核验凭据；以本文件与更新后的 `.bib` 为准。
