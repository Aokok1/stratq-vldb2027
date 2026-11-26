# StratQ · Sprint 4 独立 PC 红队模拟审稿

审计日期：2026-11-26  
审计对象：`4_LaTeX_Source/vldb_template/stratq.pdf` 对应的 12 页 Regular Research 终稿、证明包、`src/stratq/` 与冻结结果 JSON。  
性质：单代理三角色独立模拟，不是真实 PVLDB 审稿记录，也不预测录用概率。评分采用本地 1–5 分制：5=强接收，4=接收，3=边界，2=拒绝，1=强拒绝。

## 0. 执行摘要与改前评分

| 视角 | 改前评分 | 判断 | 核心理由 |
|---|---:|---|---|
| Reviewer 1：DB 查询优化与 AQP | **3/5** | Borderline reject | 三项 Scope 均能满足，系统与 SQL 契约有新意；但工程描述强于实现，经典/现代对照的契约差异需要更显式，Yelp COUNT 失利的机制解释尚无直接证据。 |
| Reviewer 2：序贯统计、SAVI、数值严谨性 | **2/5** | Reject in current form | 主定理思路成立，TeX 的 $\ell^+$ 方向正确；但证明包下端点写成 $\ell^-$，正文重启规则与 Theorem 2/代码不一致，Decimal log enclosure 与一般非 binary 归一化的证明链仍需补严。 |
| Reviewer 3：实验、统计、复现 | **3/5** | Borderline reject | 负载与系统评测广；但 3-seed 结果缺离散度，10M 表把阶段总墙钟写成纯查询墙钟，基线决策并非同路由比较，Artifact 缺 GPU 输入自动获取。 |
| Meta-Reviewer | **2.5/5** | Weak reject before correction | 贡献应聚焦“数据库执行契约 + 可预测重拟合 + 数值安全”，而非统计部件新颖性。上述问题大多可快速闭环，但当前文字/证明/实现不一致会实质伤害可信度。 |

**总判断：改前不建议直接提交；完成下列 Must-Fix 后可进入作者终检，而不是自动宣称可录用。**

---

## 1. Reviewer 1 · 数据库查询优化与 AQP 专家

### 1.1 PVLDB Scope Policy 三项硬指标

官方政策要求逐项审查 Core Data Management Focus、Connection to DB Literature、Evaluation in DM Contexts，而非凭关键词判定。[VLDB 2027 Submission Guidelines](https://vldb.org/2027/submission-guidelines.html)

| 硬指标 | 判定 | 依据 | 仍须修正 |
|---|---|---|---|
| Core Data Management Focus | **通过** | 问题是带昂贵谓词的 `GROUP BY COUNT/HAVING/TOP-k`；贡献包含列式 profiling、跨组路由、停止/弃权、DuckDB+Arrow 执行与 10M 行开销。 | “exact SQL semantics/database-native”只能指查询契约与 API，不能暗示已实现 SQL 解析、优化器或计划集成。 |
| Connection to DB Literature | **通过** | BlazeIt、ABae、LOTUS、PilotDB、BARGAIN、OLA、VerdictDB、LWS、Congressional Sampling 等 DB 文献均进入方法定位或评测。 | 对照表是论文能力矩阵，不是实测结果；LOTUS 等未实测项目必须明确为 citation-only，避免读者误以为 11 个系统均已端到端复现。 |
| Evaluation in DM Contexts | **通过** | 6 个负载族覆盖 GPU 文本/视觉、CPU 语义谓词、表格回放与 Parquet 扩展；报告调用数、冷/热延迟、覆盖率、内存与决策认证。 | 必须公开代理训练/评分是否计入“首查询”延迟，并纠正 10M 内存表口径。 |

### 1.2 与最近邻工作的切割

| 工作 | 当前切割是否成立 | 红队意见 |
|---|---|---|
| BlazeIt (PVLDB'19) | **基本成立** | BlazeIt 的 always-valid EB 与控制变量是祖先组件；StratQ 的差异是固定有限总体、组内均匀 WoR、跨组单 filtration、逐组契约与决策认证。不能说 StratQ 发明了 anytime 控制变量。 |
| ABae (PVLDB'21) | **成立但比较弱** | ABae-inspired 只是两阶段代理分层固定预算 adaptation，改变了抽样、估计量与区间，不能据此证明对所有 ABae 部署的普遍优势。正文已有“inspired/operator-level”，应继续保持。 |
| LOTUS (PVLDB'25) | **契约切割成立，未实测** | LOTUS 的 gold-relative 语义准确率/固定样本修正不等于组级 anytime count。当前没有 LOTUS 实验，不应让“11 comparators”的措辞掩盖这一点。 |
| PilotDB (SIGMOD'25) | **方向成立，细节待官方原文核验** | 固定样本/CLT 与 anytime 的区分成立；关于 `GROUP BY COUNT(*)` 排除范围必须逐字以正式论文/作者稿核实，不能以二手摘要断言。 |
| BARGAIN (PACMMOD'25) | **成立，但 0/30 易被误读** | 本地 `BARGAIN-Global` 只实现全局目标 adaptation，不是 LLM cascade 复刻。0/30 表示“未满足逐组覆盖准则”，**不是**证明 BARGAIN 自身全局保证失败。 |
| Spertus / Hait / CELEUS | **统计切割基本成立** | 详见 Reviewer 2；不得把固定代理恒等式、单组 WoR 或跨层选择单独称为新颖。 |

### 1.3 Yelp COUNT：4,349 vs 3,859 的解释

当前解释是“星级已强分层，TF-IDF 组内剩余方差小，部分批次 $R_k>1$ 的范围惩罚反超方差缩减”。这个解释**合理但尚未被冻结 JSON 直接证明**：JSON 只有最终调用数与区间，不含 gate reason、$R_k$、LOO variance 或成本序列。

Must-Fix 不是撤回数字，而是：

1. 增加一个 CPU-only、固定种子的 Yelp 标签回放诊断，直接记录每批 `selected_kappa/gate_reason/R_k/v_k/C_k/C_0`；
2. 若诊断支持，则将“解释”改成带诊断比例的限定句；若不支持，改为“一个待验证机制是……，当前结果只证明 Safe-LOO 不保证每个负载都省调用”。
3. 保留 Adult $+0.93\%$ 与 Yelp 负结果，不能删掉不利证据。

### 1.4 架构、Algorithm 1 与 DuckDB+Arrow 工程一致性

- **MF：单次扫描表述错误。** `engine.py` 先执行 group aggregate，再执行列裁剪投影，共两次 DuckDB 查询；不是“single DuckDB SQL pass”。应改为“一次 profiling 阶段，包含聚合查询与列裁剪投影”。
- **MF：NumPy zero-copy 表述错误。** DuckDB→Arrow 交接是 zero-copy；随后 `to_pylist`、`astype(float64)`、`column_stack` 会物化/复制。应明确零拷贝边界。
- **MF：Algorithm 1 pilot 预测器错误。** 伪代码写 $\theta=0$，代码使用 raw proxy 初始化 $\theta=(0,1,\ldots)$。必须同步。
- **MF：Safe-LOO 保证措辞过强。** “eliminate weak-proxy degradation”与“worst-case near zero”没有证明，Adult/Yelp 已出现轻微退化。改为“LOO 成本代理未超过基线 5% 时回退；不承诺逐查询最坏调用量”。
- **MF：Higham 总量误差上界写大了一个 $N_g$。** 对 $[0,1]$ 特征，实际使用 $\gamma_{N_g}\sum_i|\phi_i|\le\gamma_{N_g}N_g$，不是正文的 $\gamma_{N_g}N_g^2$。
- 组索引使用 stable `argsort`，正文 $O(N\log N)$ 正确；代码注释中的“O(N) grouping”应同步，避免公开复现者混淆。
- “embeddable engine”应明确用户已提供 group key、proxy/covariate 列与 callback；当前不是通用 SQL 前端。

### 1.5 Reviewer 1 结论

**Soundness 3/5；Novelty 3/5；DB relevance 5/5；Clarity 3/5；Reproducibility 3/5。**  
主贡献有 DB 价值，但当前版本把若干工程事实和统计保证写得比实现更满。完成 MF 后，我会给到边界接收，而不是仅因“ML 味较重”拒绝。

---

## 2. Reviewer 2 · 序贯统计、SAVI 与数值严谨性专家

### 2.1 `stratq_method.tex` ↔ `PROOF_PACKAGE.md` ↔ `bounds.py` 逐行核对

| 组件 | 数学核对 | 文档/代码一致性 | 判定 |
|---|---|---|---|
| $Z=\bar f+y-f(i)$、$a=\bar f-f_{\max}$、$R=1+f_{\max}-f_{\min}$、$X=(Z-a)/R$ | 条件均匀下 $\mathbb E[Z]=(K-S)/m=w\mu-h$，$X\in[0,1]$ | 三处一致 | 通过 |
| $A=\sum\lambda w/R$、$B=\sum\lambda(Z+h)/R$、$V=\sum\psi(\lambda)(X-p)^2$ | 对数 e-process 为 $\sigma(B-A\mu)-V$ | 三处一致 | 通过 |
| Ville + 两尾 + 跨组 union bound | 每组每尾 $\alpha_g/2$，总计 $\sum_g\alpha_g\le\delta$ | 一致 | 通过 |
| Theorem 1 单全局 filtration | route 与 predictor 均 $\mathcal F_{t-1}$-可测；批内固定 predictor，逐条更新 | 代码先 update CS 再写 history，支持该论证 | 通过 |
| Proposition 1 固定代理恒等式 | $\hat S_s=(S+m\bar f)/N+m(y-f)/N=(S+mZ_s)/N$，条件均值 $\mu$ | 与 CELEUS uniform-WoR 信号一致；不是新估计量 | 通过，应保留“不新颖”声明 |
| Safe-LOO Hat/LOO | $H=J/n+Z(Z^\top Z+\Lambda)^{-1}Z^\top$ 对应未惩罚截距 Ridge；$e=(r-\bar r-Z\beta)/(1-h_{ii})$ | 代码 `ridge_loo_residuals` 与公式一致 | 通过 |
| Safe-LOO 成本代理 | $\lambda=\min(.5,\max(10^{-4},\epsilon R/v))$；$C=\log(2/\alpha)R/[\lambda(\epsilon-\psi v/(\lambda R))]$ | `_betting_cost` 一致；fallback 是启发式，不参与覆盖证明 | 通过 |
| “out-of-sample variance estimate” | LOO 残差方差是预测/选择代理，不是无偏方差定理 | 正文措辞需降级为 “LOO residual-variance surrogate” | **MF（措辞）** |
| Eq. 13 下端点 | 保守下端点必须减 $\ell^+$，因为要扣除 log threshold 的上界 | **TeX 用 $\ell^+$，代码 penalty interval 也用上界** | TeX/代码正确 |
| `PROOF_PACKAGE.md` L79 | 当前写 $B^- -(\ell^-+V^+)$ | **错：$\ell^-$ 可能小于真值，会低估罚项、抬高下端点** | **MF（证明包方向性错误）** |
| 空交集重启 | 安全重启只能由 previous 与 deterministic 保守不交触发；三路保守空交集应取凸包 | 正文更新式仍写“any empty → restart D”，Theorem 2 与代码则是受限重启/凸包 | **MF（正文自相矛盾）** |
| `bounds.py` L805 注释 | 注释称保守三路空交集“证明 exact intersection empty” | 错误：保守加宽本身可能制造空交集；凸包输出正确，但理由应改为“不丢弃真实交集” | **MF（代码注释）** |
| $\psi$ 级数与尾项 | 保留 $k=2,\ldots,97$ 共 96 项；$\lambda^{98}/[98(1-\lambda)]$ 在 $\lambda\le.5$ 时 $<2^{-100}$ | 代码一致 | 通过 |
| $\log(2/\alpha)$ 的 Decimal enclosure | 80 位 `Decimal.ln` 正确舍入后直接转 float，再各 `nextafter` 一步，缺少“十进制舍入误差跨 float 中点”的严格边界 | 当前测试未覆盖这一证明细节 | **MF（严格 enclosure）** |

### 2.2 非二进制安全路径的证明缺口与事件边界

论文主契约是 binary predicate，现有六组实验也都用 0/1 标签。公开 engine 同时广告 `mean/sum`，`WoRControlVariateCS` 接受任意 `y_min/y_max`，因此安全路径必须覆盖一般有限 binary64 标签。

初审发现 `_update_batch_safe` 将 `_normalize_y(...)` 的区间只取 `[0]`，没有把归一化标签的完整向外区间传给后续算术。这不只是一般 `mean/sum` 路径问题：对 binary label 1，directed division 返回的下端也可能是 `nextafter(1,-inf)`，旧实现同样只保留下端。后续 `nextafter` 加宽可能在受测路径上补救，但不能替代“每个基本操作均 enclosure”的论证。

必须区分两类检验：

- **覆盖事件检验：** 单条 CS 路径排除真值可能发生在允许的 $1-\delta$ 失败事件上，不能冒充浮点反例。
- **数值包络检验：** 只有 safe 区间未包含同参数、同 betting/lambda 的高精度 Decimal CS 区间，才是浮点实现错误。

曾构造 $N=66$、所有 raw label 为 `0.2`、$y_{\max}=0.3$ 的路径，第 61 步 safe 与 Decimal CS 都排除真值；进一步确认这属于真实算术失败事件，不是浮点反例。旧实现 5,000 条随机非 binary 路径与 Decimal 区间对照也未发现实际包络失败，但证明缺口仍然存在。

因此 Must-Fix 的准确定性位为：

1. 全区间参与归一化标签算术，消除证明缺口；
2. 新增确定性 Decimal 对照与 300 条随机非 binary 路径回归；
3. 对受测 Adult 回放核对 calls/decisions 不变，端点允许向外放宽少量 ULP；所有论文报告数字按原 JSON 重新校验；
4. 不把“单路径排除真值”误报为“浮点不安全”。

### 2.3 与 Spertus / CELEUS / Hait / Rank CS 的过度声称检查

- **Spertus et al. (2026)：** 稿件已承认其跨层选择、WoR remaining mean、simultaneous LCB；StratQ 只声称在其上加入可预测重拟合控制变量与 SQL 执行。矩阵“Refit CV=x”应限定为“row-level batch-refitted CV not provided”，避免否认其所有 plug-in/augmentation。
- **CELEUS (2026)：** Proposition 1 明确固定代理 uniform-WoR 恒等式不新颖，正确。不得把“IPW 方差 blowup”作为 CELEUS 的一般结论；其非均匀设计有 probability floor，当前只支持 uniform 特例。
- **Hait (2026)：** i.i.d. within-stratum 与固定总体 WoR 的切割成立；其 target-set/best-system 认证与 SQL top-k 不同，但“single joint event”本身不新，稿件已承认。
- **Rank CS (2026)：** anytime leaderboard 已存在，稿件明确承认；StratQ 只应声称固定快照组计数、WoR 与关系执行成本下的契约。
- **SAVI/WSR：** 采用而非替换 WSR betting engine 的表述正确。

### 2.4 Reviewer 2 结论

**Soundness 2/5；Novelty 3/5；Rigor 2/5；Clarity 3/5。**  
Theorem 1 没有发现原则性反例；问题集中在文档方向符号、正文/Theorem 2 不一致，以及公开非 binary 安全路径的证明缺口。修复前不能以“zero-error compile/47 claims passed”替代数学验收。

---

## 3. Reviewer 3 · 实验设计、统计严谨性与可复现性专家

### 3.1 实验口径

- **6 个负载族：** Yelp、CIFAR-10、Fashion-MNIST、SMS、Adult、Synthetic Scale/Stress。把 Synthetic 三项研究算作一个族可接受，但需明确它不是单一 workload。
- **11 个 comparators：** 5 个 modern + 6 个 classical database operators。正文另外报告 independent grouped hypergeometric CS；应写“11 个数据库/现代 operator adaptations，另加一个独立 hypergeometric CS”，否则实际外部方法数量会被误读为 12。
- **operator-level adaptation：** 总体声明一致；所有 BARGAIN/PilotDB/CELEUS/Spertus 结果不得删掉“not full replica”。LOTUS 未实测，不能放进实测 comparator 计数。
- **COUNT 与 decision 不是同路由比较：** StratQ 的 HAVING/TOP-2 使用专用边界路由，其他现代基线由 COUNT 最终区间派生。Table 1 已说明，但正文“beats/fails”必须反复限定为“在该派生协议下”，不能表述成原始系统的最佳决策性能。
- **BARGAIN：** 0/30 是逐组 count 准则失败，不是其全局契约失败；图 3/正文需要同句澄清。
- **固定预算经典对照：** 同时改变 allocation、replacement、interval 和 proxy fitting，只能支持“这些 adaptations 的表现”，不能归因于单一组件。

### 3.2 种子、离散度与统计边界

- Yelp/CIFAR COUNT 与 decision 均只有 **3 matched seeds**；3/3 认证是描述性结果，不能写成稳定 100% 保证。
- Proxy sweep 每格 3 seeds；Zipf 是 **15 个 $(G,\alpha)$ cells × 2 seeds = 30 runs/method**，不是“30 settings/cells”。
- Adult 30 seeds、SMS 10、Fashion 5 更充分，但仍是固定 snapshot 上的经验覆盖，不是跨总体证明。
- **MF：** 对关键 3-seed 调用量补 sample SD 或 min–max，并明确 $n=3$、不做显著性/录用推断。离散度可直接从现有 JSON 计算，不需重跑 GPU。
- **MF：** 把“Safe-LOO eliminates degradation/bounds worst-case near zero”改为 empirical observation；Adult $+0.93\%$、Yelp COUNT 与 SMS 慢速边界必须保留。

### 3.3 表/图/正文数字口径

- **MF：10M 内存表 Wall 列错配。** `sprint2_memory_10m.json` 的 query 阶段 `wall_ms=187,849.93`（含重新建 profile），`query_ms=37,857.28` 才是纯查询。Table 4 当前写 37.86 s 却标“Online Query (incl. Prof.) / Wall*”，自相矛盾。应拆成阶段总墙钟与纯查询墙钟。
- **MF：+21.69 MB 的时间点需写清。** 该值在查询结束后、`tracemalloc` teardown 前测；表内 End RSS 是 teardown 后最终值，不能直接拿两者相减。应标明“post-query, pre-teardown RSS delta”。
- **MF：Fig. 4(b) “only StratQ holds valid anytime intervals”过宽。** 独立 grouped hypergeometric CS 也是 anytime；应改为“图中这些 fixed-look/asymptotic arms 中，仅 StratQ 提供 finite-sample anytime 契约”。
- **待核：Fashion 3.36% vs Table 3 3.59%。** 前者若为 mean per-group half-width、后者若为 Max HW，必须分别标清；若是同一指标，则数字错配。
- **MF：960 decision trials 设计应写全。** $4$ gaps × $3$ budgets × $20$ seeds × proxy on/off × $2$ queries = 960；当前段落只列四乘三乘二十，漏掉两个二元因子。
- **首查询延迟边界：** GPU fresh 计时包含 profile + query + reference inference，但 proxy model training/scoring 与数据加载在此之前完成。必须显式声明，否则“end-to-end from first query”会被理解为包含模型准备。
- **Claim audit 覆盖不足：** 47/47 证明注册数字匹配，不证明未注册正文数字、图件、公式、统计解释或所有 JSON 字段正确。Sprint 4 应扩展 SD/设计因子/10M wall 字段检查。

### 3.4 Artifact 与复现

- 当前 `artifact_tools.py fetch-data` 只覆盖 Adult/SMS/Fashion；**不获取 Yelp、CIFAR-10、DistilBERT-SST2 或 ResNet34 backbone**，而 GPU runner 直接依赖本地文件。全新机器无法复现 Sprint 2，这是 Must-Fix。
- 发布包 PATTERNS 未包含 `figure_metrics_sprint3.json` 与 `PAPER_CLAIM_AUDIT_SPRINT3.json`，却打包旧版 claim audit；必须修正。
- 核心离线验证应使用 `pytest` 与 Sprint 3/4 claim audit；旧 `unittest discover` 不能覆盖 pytest 风格测试。
- requirements 缺 `pytest`、`PyMuPDF`；GPU optional dependencies 与固定模型/数据 revision/SHA-256 必须写入 `ARTIFACT.md`。
- CIFAR reference model 每轮重训，GPU/bf16 训练不保证跨硬件 bitwise 相同；应记录 backbone/data 哈希，并明确微调模型哈希只是本次测量指纹，不是可重复输出承诺。

### 3.5 Reviewer 3 结论

**Experimental breadth 4/5；Statistical reporting 3/5；Internal consistency 2/5；Reproducibility 2/5。**  
无需重跑数小时 GPU 就能修复大部分问题：从现有 JSON 计算 SD、修正标签、补充设计因子、打通 artifact 获取链并做一次 CPU gate diagnostic。

---

## 4. Meta-Reviewer

### 4.1 贡献是否足以支撑 Regular Research

最强的一句话贡献是：

> 在固定快照的昂贵二元谓词上，把可预测批间重拟合控制变量、组内均匀 WoR、跨组单 filtration 与定向 binary64 包络统一为一个可直接认证 COUNT/HAVING/TOP-k 或弃权的 DuckDB+Arrow 执行契约。

这不是新 concentration inequality，也不是“第一次 anytime top-k”。其价值在系统契约与实现闭环。Intro/Abstract 应持续聚焦这一点，避免统计新颖性过度声明。

### 4.2 改前主要顾虑

1. 审稿人无法确认证明包、正文与代码讲的是同一程序；
2. 3-seed 与派生决策比较容易被读成过强系统优越性；
3. 公开 Artifact 当前不能从空目录恢复 GPU 输入；
4. 若干“零误差/47/47”是结构门禁，不是独立统计与复现验证。

### 4.3 必须关闭的 Must-Fix 清单

| ID | 级别 | 问题 | 关闭标准 |
|---|---|---|---|
| MF-01 | Must-Fix | `PROOF_PACKAGE.md` 下端点使用 $\ell^-$ | 改为 $\ell^+$；新增文档/代码方向性检查 |
| MF-02 | Must-Fix | 正文空交集重启规则与 Theorem 2/代码不一致 | 正文改为受限重启 + 三路保守碰撞凸包；代码注释同步 |
| MF-03 | Must-Fix | 归一化标签只取区间下端（binary 1 也可能为 nextafter(1,-inf)），证明链不完整 | 全区间算术；Decimal 回归通过；calls/decisions 不变，端点允许向外放宽 ULP |
| MF-04 | Must-Fix | Decimal log 转 binary64 的 enclosure 证明缺一步 | 显式包住 Decimal 正确舍入误差后再 outward conversion；测试通过 |
| MF-05 | Must-Fix | Safe-LOO “eliminate/worst-case near zero”过度声称 | 改为 5% LOO 成本门控与 empirical overhead；同步 Abstract/CMT |
| MF-06 | Must-Fix | 单次扫描、NumPy zero-copy、pilot $\theta=0$ 与代码不符 | 按实现修正系统段与 Algorithm 1 |
| MF-07 | Must-Fix | Higham 特征总量误差上界写为 $\gamma_NN^2$ | 改为 $\gamma_N\sum|\phi|\le\gamma_NN$ |
| MF-08 | Must-Fix | Yelp COUNT 机制解释无直接证据 | CPU gate diagnostic 落盘；按结果限定或撤回机制断言 |
| MF-09 | Must-Fix | 3-seed 无离散度；15×2 被写成 30 settings；960 设计因子不全；comparator 计数含糊 | 补 SD/range、明确 n 与试验设计、区分 11 adaptations + hypergeom |
| MF-10 | Must-Fix | 10M query 阶段 wall 与 query_ms 错配；+21.69 MB 时间点不清 | 表/文/图使用相同字段名并拆阶段总墙钟/纯查询 |
| MF-11 | Must-Fix | Fig. 4(b) “only StratQ anytime”忽略 hypergeom | 限定为图中 fixed/asymptotic arms |
| MF-12 | Must-Fix | BARGAIN 0/30 与 decision 比较易被过度解读；GPU 首查询计时边界不清 | 同句披露 global target、COUNT-derived decisions、proxy/setup exclusion |
| MF-13 | Must-Fix | Artifact 缺 GPU 数据/模型自动获取与新版审核文件 | 固定 revision + SHA-256 获取；打包新 manifest/claim audit；空目录离线门禁通过 |
| MF-14 | Must-Fix | 62 文献正式状态/撤稿/上下文审计尚未闭环 | 以官方记录更新 `BIBLIOGRAPHY_AUDIT.md` 与必要 BibTeX；无证据项保持 preprint |
| MF-15 | Must-Fix | Table 1 比较轴与 ABae/BARGAIN/CELEUS/Hait/Spertus/LOTUS 单元格误标；PilotDB 排除项写窄；BlazeIt 被误称 fixed-look | 按官方原文重定义轴并修单元格；PilotDB 仅排除 aggregate 作为 GROUP BY key；本地 BlazeIt arm 改称 inspired fixed-look |
| MF-16 | Must-Fix | `+21.69 MB` 是单次瞬时 RSS 差，不是 retained/query-only memory；阶段末差为 −0.68 MB，query 子进程峰值含 profile | 摘要/结论/Fig. 5 改为瞬时观测 + 85.6 MB traced peak + −0.68 MB end-RSS delta；禁止“only/净增量”措辞 |
| MF-17 | Must-Fix | GPU fresh 未清 ψ cache；延迟容差为 3/4/5% schedule；proxy training/scan 未计入；COUNT wall 不能复算 speedup | 明确 precomputed-proxy/warm-cache/fresh-profile 边界、容差与 seeds；从 JSON 报告含 proxy scan 的 sensitivity |
| MF-18 | Must-Fix | HAVING/TOP-2 baseline 是 post-hoc COUNT-derived，StratQ 是专用 routing，非等预算决策比较 | 全文限定为 decision-aware routing vs post-hoc protocol；CIFAR HAVING τ=979 写入 |
| MF-19 | Must-Fix | BARGAIN JSON 未记录 global truth/interval/coverage，0/30 不能评价其原全局保证 | 删除“global baselines fail/全败”；只报告 proportional per-group plug-in 的逐组覆盖 |
| MF-20 | Must-Fix | PilotDB tail coverage 94.7% 错，原始为 215/222=96.85% | 修正报告/图注；optional-stop 明确为 fixed-look CLT 诊断误用，不是原协议失败 |
| MF-21 | Must-Fix | 11 comparators 实际漏算 independent hypergeometric CS；Calibrated 是独立 control | 改为 11 adaptations + 1 independent CS，或明确 12 个非 StratQ 方法；six workload families |
| MF-22 | Must-Fix | CPU live_latency 与 Sprint 1 exhaustive timing 数量级冲突，Fashion 胜负结论不稳 | 对齐协议重跑轻量 timing 或撤回跨研究混合比较；不得保留无解释的 646 vs 1961 |
| MF-23 | Must-Fix | SMS 表省略 legacy，但 safe_loo 实际不优于 legacy；Max HW 是 per-run worst-group 的均值 | 补 legacy 行或改写叙述；统一 metric 名，3.36% mean normalized HW 与 3.59% mean Max HW 分开 |
| MF-24 | Must-Fix | 47/47 audit 漏检 raw SD、决策正确性、tail 分母、routing 180/180、memory 语义、hypergeom | 扩展 Sprint 4 claim audit，所有新增数字/口径来自 JSON 重算 |

### 4.4 Polish 清单

| ID | 建议 | 理由 |
|---|---|---|
| P-01 | Table 1 每个 x/pmark 增加一行脚注限定“论文能力，不是本地实测” | 防止矩阵被误读为系统复现表 |
| P-02 | 将 “database-native/exact SQL” 改为 “database-facing execution contract/API” | 当前没有 SQL parser/planner |
| P-03 | Fig. 2–5 caption 直接写 seed 数与 mean±SD 口径 | 提高图的自包含性 |
| P-04 | 给 Safe-LOO 门控补一个失败案例小表（n<12、df<3、span>1、no improvement） |  reviewers 可快速核对实现 |
| P-05 | 对 PilotDB optional-stop 明确标注“diagnostic misuse of fixed-look CLT” | 0/6 不是原系统在正确使用下的失败率 |
| P-06 | 压缩重复 related-work 句子，把空间用于统计口径脚注 | 保持 ≤12 页 |
| P-07 | 统一 “reused/warm/fresh profile” 术语 | 避免 warm query 与 warm cache 混淆 |
| P-08 | 公开材料中删除内部审稿过程与旧 archive 作为主入口，只保留 RC1 + 审核记录 | 降低复现者选择成本 |

---

## 5. Round 2 复审结论

复审日期：2026-11-26。Round 2 以“实际文件/JSON/测试是否改变”为准，不以文字声明关闭。

### 5.1 Must-Fix 逐条关闭

| ID | 状态 | 关闭证据 |
|---|---|---|
| MF-01 | 已关闭 | `PROOF_PACKAGE.md` 下端点改为 $B^--(\ell^++V^+)$；`audit_bibliography_sprint4.py` 与 88 项审计均检查方向。 |
| MF-02 | 已关闭 | `stratq_method.tex` 改为“previous 与 deterministic 保守不交才 restart；三路碰撞取凸包”；`bounds.py` 注释不再声称保守碰撞证明 exact empty。 |
| MF-03 | 已关闭 | `_normalize_y` 全区间进入算术；新增 600 个 log-threshold Decimal 检查、确定性非 binary replay、300 条随机非 binary 路径；全套 52 tests 通过。Adult calls/decisions 不变，端点仅向外放宽。 |
| MF-04 | 已关闭 | Decimal log 先取相邻十进制端点再 outward conversion；140-digit Decimal 对照通过。 |
| MF-05 | 已关闭 | Abstract/Intro/Method/Limitations/CMT 均改为“5% LOO 成本门控”，删除 eliminate/worst-case near-zero；Yelp/Adult 退化如实保留。 |
| MF-06 | 已关闭 | 系统段改为一次 profiling stage（aggregate + projection）；明确 DuckDB→Arrow 与 NumPy 物化边界；Algorithm 1 pilot 改为 raw-proxy $\theta=(0,1,0,\ldots)$。 |
| MF-07 | 已关闭 | Higham 特征总量上界改为 $\gamma_{N_g}\sum|\phi_i|\le\gamma_{N_g}N_g$。 |
| MF-08 | 已关闭 | `sprint4_yelp_gate_diagnostic.json`：202 gates、112 selected、75 no-improvement、15 insufficient；LOO variance −12.9%，span ≤8.9e−16，无 R>1.01。原范围惩罚解释撤回，正文改为负结果。 |
| MF-09 | 已关闭 | `sprint4_summary_statistics.json` 从 raw runs 重算 SD/range；正文补 3-seed COUNT SD、15×2 Zipf、960 设计因子、11 adaptations + independent CS。 |
| MF-10 | 已关闭 | Table 4 拆 query subprocess 187.85 s / profile rebuild 103.50 s / query-only 37.86 s；claim audit 直接检查 JSON 两字段。 |
| MF-11 | 已关闭 | Fig. 4(b) caption 限定为图中六个 fixed/asymptotic adaptations，并明确 independent hypergeometric CS 也是 anytime。 |
| MF-12 | 已关闭 | 全文区分 dedicated routing 与 post-hoc COUNT-derived protocol；CIFAR HAVING $\tau=979$ 写入；删除“等预算 baseline 也无法认证”的暗示。 |
| MF-13 | 已关闭 | `artifact_tools.py` 补齐全部固定输入获取、CMT/新审计文件与 requirements；RC1 全新临时目录解压后 manifest、52 tests、文献/CMT/88 项审计全通过，临时目录已清理。 |
| MF-14 | 已关闭 | 62 条在线审计落盘；3 条元数据升级；8 篇 preprint 保持；Spertus 非录用；撤稿筛查限制如实写明。 |
| MF-15 | 已关闭 | Table 1 重定义轴并修正 ABae/BARGAIN/CELEUS/Hait/Spertus/LOTUS；PilotDB 仅排除 aggregate 作为 GROUP BY key；BlazeIt 本地 arm 改称 inspired fixed-look。 |
| MF-16 | 已关闭 | 摘要/结论/Fig. 5/Table 4 删除“only +21.7 MB incremental/retained RSS”；改为即时 +21.69 MB、traced peak 85.6 MB、final −0.68 MB 的单次观测。 |
| MF-17 | 已关闭 | 正文公开 precomputed proxy、warm model/cache、3/4/5% tolerance、fresh 仅重建 profile、冻结运行共享 ψ cache；补 Yelp scan/training sensitivity。未来 runner 会清 fresh ψ cache 并记录设置。 |
| MF-18 | 已关闭 | 与 MF-12 同源：HAVING/TOP-2 结果句均限定 post-hoc protocol，CIFAR τ 已补。 |
| MF-19 | 已关闭 | 删除“global baselines fail/全败”；明确 BARGAIN JSON 无 global interval，0/30 仅是 surrogate 的逐组 plug-in 准则。 |
| MF-20 | 已关闭 | PilotDB tail coverage 修正为 215/222=96.85%；optional-stop 明确为诊断协议误用。 |
| MF-21 | 已关闭 | 全文统一 six workload families、11 database/modern adaptations + independent hypergeometric CS（12 个非 StratQ 方法）。 |
| MF-22 | 已关闭 | `sprint4_cpu_latency_aligned.json` 用同一模型/行/随机 blocks 重跑；正文改为 Fashion exact 1430.5 ms vs fresh 577.3/reused 423.5，SMS exact 519.2 vs fresh 902.3/reused 834.7。 |
| MF-23 | 已关闭 | Table 3 补 SMS legacy 三行；正文明确 safe_loo 不优于 legacy；Max HW 定义为 per-run worst-group half-width 的均值，3.36% mean normalized HW 与 3.59% Max HW 分开。 |
| MF-24 | 已关闭 | claim audit 扩展为 88 项，直接检查 raw SD、decision correctness、tail 分母、180/180 routing、memory 语义、hypergeom、proof/bib/CMT/PDF/LaTeX；88/88 通过。 |

### 5.2 改后评分

| 视角 | 改后评分 | 判断 |
|---|---:|---|
| Reviewer 1 | **3.5/5** | Borderline accept：DB scope 清楚，工程描述与实现一致，比较边界诚实；仍缺全系统复刻与 SQL parser/planner 集成。 |
| Reviewer 2 | **3.5/5** | Borderline accept：主定理无原则性反例，证明/公式/代码方向一致，数值回归充分；Lean/Coq 形式化与独立外部证明审阅仍是有价值的增强。 |
| Reviewer 3 | **3.5/5** | Borderline accept：统计离散度、试验设计、memory/timing 口径已修正；3-seed/2-seed 与 post-hoc 决策比较仍是明确局限。 |
| Meta-Reviewer | **3.5/5** | Borderline accept：贡献聚焦数据库执行契约，不再靠过度归因；Artifact 独立解压门禁已通过，最终提交仍受四项作者人工开关约束。 |

### 5.3 Polish 项处置

- Table 1 已加“capability matrix, not full replicas”与轴定义。
- “database-native/exact SQL”已改为 database-facing execution contract/API。
- Fig. 2–5 caption 已补 seed/setup/protocol 边界；Fig. 5 内部标注改为 final ΔRSS −0.68 MB。
- Safe-LOO 失败门控与 Yelp trace 已公开；完整逐批 trace 在 Artifact JSON。
- PilotDB optional-stop 已标 diagnostic mismatch；BARGAIN 已标 global-count surrogate。
- Related work 已压缩以保持 12 页，同时保留全部 62 个引用 key。
- fresh/reused 术语已统一为 TableProfile 边界；cache/setup 单列。
- RC1 将只作为公开候选入口，旧 review ZIP 不再作为主推荐。

**复审总判断：全部技术 Must-Fix（含 Artifact 独立解压门禁）已关闭；四项作者人工开关完成后，可进入 CMT 提交前最终确认。此评估不是录用概率，也不替代真实 PC/Meta-Reviewer。**
