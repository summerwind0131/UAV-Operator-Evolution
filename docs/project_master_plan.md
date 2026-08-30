# Trajectory-Informed Operator Evolution 项目总体实施计划

状态：实施中  
计划基线日期：2026-08-29  
研究主线：从 UAV 单领域实现提炼通用核心，以 JSSP 验证通用性，再开展机制级双向迁移

## 1. 里程碑总览

| 里程碑 | 状态 | 检查点 / 完成门 |
| --- | --- | --- |
| Step 0：领域边界与 golden characterization | 已完成 | `d44545f`，完整基线 `290 passed, 3 skipped` |
| Step 1：`InstanceRef`、`ObjectiveEvaluation` 与 UAV 纯适配 | 已完成 | `d44545f` |
| Step 2：完整 UAV `DomainAdapter` | 已完成 | `eda6987`，标签 `phase1-step2` |
| Hidden Test-v2 最终评价收口 | 已完成 | `a8be8f6`、`uav2d-hidden-test-v2-final-v1`、GitHub Release |
| Step 3：通用搜索内核 | 已完成 | `39a835b` + UAV routing commit；shadow/RNG identity 全绿 |
| Step 4：通用轨迹、诊断与 UAV snapshot | 已完成 | core trace/recorder/diagnoser；SQLite、JSONL、identity 保持一致 |
| Step 5：通用候选验证与确定性策略 | 已完成 | UAV retention 逐字段一致；`deterministic-v2` 与隔离性能任务已建立 |
| Step 6：通用提案信封与领域 IR | 已完成 | `proposal-envelope-v1`、`UAVDomainKit`、旧 hash 兼容与 fail-closed 已验收 |
| Step 7：通用演化管理器依赖注入 | 已完成 | split freeze、依赖注入、全套 UAV gate 与性能门通过 |
| JSSP 第二领域验证 | 已完成 | 正式 400/240/400、3×3、60/41/41 qualification 完成；标签 `cross-domain-core-qualification-v1` |
| 单仓三包与 core `0.1.0` | 已完成 | 标签 `trajectory-core-v0.1.0`、独立 wheel/sdist、GitHub prerelease、commit/hash receipt |
| UAV↔JSSP 机制迁移 | 未开始 | 双向三臂实验、封存测试、统计与复现包 |

本文件是项目的唯一总体路线图。每次里程碑完成后更新状态、提交、测试结果和 artifact receipt；历史实验和 frozen artifact 不得覆盖。

## 2. 总体路线与实施顺序

1. 固化 Step 0–2 和已经完成的 UAV Hidden Test-v2。
2. 完成通用化第一阶段 Step 3–7，要求 UAV 行为与结果不变。
3. 在当前仓库接入 JSSP，以结构明显不同的离散约束问题验证核心。
4. 验收后在现有仓库内固化 core/UAV/JSSP 三包边界，并发布 core `0.1.0` GitHub prerelease。
5. 开展 UAV↔JSSP 双向机制迁移研究。
6. 形成最终技术报告、复现包和版本化研究结论。

每个 Step 单独提交。定向测试、完整测试、`git diff --check` 和工作树审计通过后再推送。

## 3. 第一阶段：完成 UAV 零行为通用化

### 3.1 当前检查点与 Hidden Test-v2 收口

状态：已完成。发布页：<https://github.com/summerwind0131/UAV-Operator-Evolution/releases/tag/uav2d-hidden-test-v2-final-v1>；当前完整测试为 `291 passed, 3 skipped`。

- 推送 `d44545f` 和 `eda6987`，以 `phase1-step2` 建立检查点。
- 只核验既有 Hidden Test-v2 opening、execution、audit receipt 和 6,960 条唯一记录；终测已经完成，不再打开或重跑。
- 修正仍写着 `sealed_unrun` 的过期文档，提交审计报告、摘要和 receipt。
- 对大体积原始结果生成规范化压缩包及 SHA-256，作为 GitHub Release artifact 保存，不将原始大文件直接加入 Git 历史。
- 正式报告预注册结果：Evolutionary AFL-UAV v1 具有成本优势但可行率退化，两个关键消融产生负结果；禁止根据结果修改或重新评价 v1。
- 创建不可变标签 `uav2d-hidden-test-v2-final-v1`。

### 3.2 Step 3：通用搜索内核

状态：已完成。core 不导入 UAV 类型；UAV façade 保持原构造、返回类型、导入路径、trace identity 与四子流消费顺序。逐步 shadow hash 为 `d26477dd9d64bc581dfa4855c1a623369626403abaf98c07d95c2d7bd5f4a820`；当前完整测试为 `295 passed, 3 skipped`。

- 在 core 定义泛型 `SearchContext`、`OperatorOutcome`、`SearchOperator`、`SearchStep`、`SearchResult`、搜索预算及调度接口。
- 实现只依赖 `DomainAdapter`、operator、scheduler 和 acceptance policy 的通用搜索循环。
- UAV operator facade 将现有 `PathOperator/OperatorResult` 映射到通用协议；现有 `SearchExecutor` 构造参数、返回类型和导入路径保持不变。
- 原样保留四个 RNG 子流的生成顺序、seed 范围和消耗次数。
- 新旧循环做逐步 shadow comparison：operator ID、候选/当前/最优解 hash、目标分项、接受决定、温度、停滞状态、最终 RNG 状态均一致；仅排除墙钟时间。
- shadow 全绿后让 UAV `SearchExecutor` 调用通用内核，再删除重复旧循环。
- 独立提交：`feat: add domain-independent search kernel`、`refactor: route UAV search through generic kernel`。

### 3.3 Step 4：通用轨迹、诊断与 UAV snapshot

状态：已完成。`OperatorTrace`、SQLite/JSONL recorder、延迟奖励、operator diagnoser 和 feature catalog 已归属 core；UAV 旧模块为身份相同的兼容导入。core API 接受 `instance_id`，序列化与 SQLite 继续使用 `map_id` v1 别名。UAV `SearchExecutor` 通过 `UAVTraceEncoder` 生成三态 snapshot，Step 0 identity 不变；当前完整测试为 `298 passed, 3 skipped`。

- core 拥有通用 before/candidate/accepted 三态 trace、奖励、上下文、谱系和 recorder；领域 encoder 负责解、实例及领域特征。
- 接入现有 `UAVTraceEncoder`，路径、地图、碰撞、clearance 和目标分项保留在 UAV 层。
- core API 使用 `instance_id`；当前 SQLite 的 `map_id` 物理列作为 v1 兼容别名继续保留，不做破坏性 schema migration。
- 迁移延迟奖励和通用 diagnoser 逻辑；UAV 特征分组通过 feature catalog 注册。
- 新旧 SQLite 行、JSONL 和 identity projection 必须完全一致，时间戳、墙钟和绝对路径除外。
- 独立提交：`refactor: separate core traces from UAV snapshots`。

### 3.4 Step 5：通用候选验证与确定性策略

状态：已完成。paired outcome、CRN seed schedule、ABBA timing、bootstrap、retention gate、slot replacement 与版本化 fitness 已归属 core；core 使用 `instance_id/context_label`，同时维持 UAV v1 的 `map_id/difficulty` 物理投影。UAV validator 与 manager 显式选择 `uav-legacy-v1`，JSSP/后续研究默认使用不按墙钟排名、也不允许 runtime-only retention 的 `deterministic-v2`。默认确定性套件为 `302 passed, 3 skipped, 1 deselected`，隔离性能任务为 `1 passed, 11 deselected`。

- 抽出通用 paired outcome、CRN seed schedule、ABBA timing、bootstrap、retention gate 和 population slot replacement。
- 通用 validator 只接收 validation instances、adapter、operator population 和固定预算；API 不允许传入完整 split 字典。
- UAV `FixedBudgetCandidateValidator` 保留为兼容 facade，旧 retention 结果逐字段一致。
- 引入版本化 `FitnessPolicy`：
  - `uav-legacy-v1` 保持现有 runtime rank 行为，确保历史结果不变。
  - `deterministic-v2` 用于 JSSP 和后续研究；墙钟只报告和做性能 guard，不参与父代排序或单独触发保留。
- 将单次 1 秒墙钟测试从默认确定性套件拆出，改为隔离性能任务、多次运行和中位数判定；默认测试使用可注入时钟或评价预算。
- 独立提交：`refactor: generalize paired candidate validation`。

### 3.5 Step 6：通用提案信封与领域 IR

状态：已完成。core 新增 content-addressed `CandidateProposalEnvelope`、typed `ProposalBudgetDeclaration` 与 `DomainKit` 协议；UAV 层提供 `uav-v1` kit，统一负责 IR 解析、能力目录、编译、smoke、能力使用统计、拓扑/行为指纹及静态安全评分。旧 proposal/bundle 不新增序列化字段，读取时由兼容属性隐式绑定 `uav-path-planning-2d/uav-v1`；固定旧 bundle hash 为 `b0b45eb5…ed0d`，旧 proposal hash 为 `1b1af302…c533`。domain/version 不匹配、hash 篡改和非白名单执行载荷均 fail-closed；完整默认回归为 `307 passed, 3 skipped, 1 deselected`。

- 新增 `CandidateProposalEnvelope`：`candidate_id`、`domain_id`、`ir_version`、父代、证据引用、设计理由、预算声明和 typed payload。
- 新增 `DomainKit`：IR 解析、capability catalog、schema 校验、编译、smoke、能力使用统计、拓扑指纹和行为指纹。
- `UAVDomainKit` 包装现有 `OperatorSpec`、compiler、primitive catalog 和 smoke fixture；UAV IR 固定为 `uav-v1`。
- 新 envelope 使用 `proposal-envelope-v1` hash；旧 UAV JSON 和 EvidenceBundle hash 通过兼容投影保持原值，读取旧 artifact 时隐式补入 `uav-v1`。
- Agent、Evidence Builder、Proposal Validator 和工具 dispatcher 只能通过 `DomainKit` 访问领域能力；domain/version 不匹配时 fail-closed。
- 不建立万能 DSL，也不开放任意 Python 执行。
- 独立提交：`refactor: separate proposal envelope from UAV IR`。

### 3.6 Step 7：通用演化管理器依赖注入

状态：已完成。core 提供 `EvolutionManagerDependencies`、`PopulationSeed`、`EvolutionSplitCapabilities`、population freeze receipt、fingerprint 与 artifact sink；UAV manager 默认装配原 adapter/kit/population/validator/designer/orchestrator，同时允许整组能力注入。旧字典 split 在入口转换为显式 capability，test 只有在最终 population fingerprint 固化并由同一 split 对象签发 receipt 后才能打开。`OperatorEvolutionManager(config)`、现有 CLI/YAML 与结果模型保持有效；Step 0 golden、CLI demo、Agent/Mock Multi-Agent、candidate validation 和 planner benchmark preflight 全绿。默认回归为 `310 passed, 3 skipped, 1 deselected`；7×7 独立进程中位性能相对 `8d86411` 为 `-0.594%`，通过不超过 `+5%` 的门，receipt 位于 `artifacts/releases/uav-generalization-phase1-v1.performance.json`。

- 通用 manager 注入 `DomainAdapter`、`DomainKit`、population factory、candidate validator、designer/orchestrator 和 artifact sink。
- 默认构造仍自动装配 UAV 组件，现有 CLI、YAML 和 `OperatorEvolutionManager(config)` 调用保持有效。
- train、validation、test 使用显式能力对象隔离；test 只能在 population 冻结后打开。
- 运行 Step 0 golden characterization、完整 CLI smoke、Agent/Mock Multi-Agent、candidate validation 和规划 benchmark preflight。
- 同一 Windows/Python 环境要求 identity projection 精确相同；关键路径中位性能退化不得超过 5%。
- 创建 `uav-generalization-phase1-v1` 标签；此时第一阶段才算结束。

## 4. 第二阶段：JSSP 领域验证

状态：已完成。`JobShopInstance/JobShopSolution`、无外部求解器的确定性 schedule builder、完整 `DomainAdapter`、固定八槽 P0、`jssp-v1` typed IR/compiler、三类 baseline、OR-Library 归档与 60/41/41 split 均已验收。JSSP 复用同一个 generic search、三态 recorder、延迟奖励、diagnoser、core memory、proposal envelope、CRN/ABBA validation、`deterministic-v2` retention 和 population freeze。

注册 smoke 使用 64 calls、2 代×2 候选、2 个 validation instances、2 次 timing repetitions；生成 1,024 条 trace，四个候选中一个通过 global paired gain 门，test 未打开。smoke receipt：`artifacts/releases/cross-domain-core-qualification-v1.smoke.json`，payload SHA-256 `697e5fd8…db38`。

正式 qualification 使用 60×400 training、41×240 validation、3 代×3 候选、8 槽种群及 4 次 ABBA timing repetitions；保存 24,000 条训练 trace、8 个 operator profiles 和 64 个 sequence synergies。9 个候选全部未达到预注册保留门，最终 population 与 P0 相同。freeze receipt 签发后首次打开 41 个 test instances，P0/Pn 各 400 calls：两组可行率均为 1.0，平均 makespan 均为 1772.6585，41/41 逐实例打平，mean relative gain 0、win rate 0、tie rate 1。该零结果不重跑、不调参。formal receipt：`artifacts/releases/cross-domain-core-qualification-v1.formal.json`，payload SHA-256 `ad44cc84…2f7d`；[GitHub Release](https://github.com/summerwind0131/UAV-Operator-Evolution/releases/tag/cross-domain-core-qualification-v1) 复现包 SHA-256 `7c15634b…851b`。

### 4.1 数据、模型与确定性调度

- 先在当前仓库建立 `jssp_operator_evolution` 领域包，验证完成后再拆库。
- 训练集为 60 个固定合成实例：`6×6`、`10×10`、`20×15` 各 20 个，主 seed `20260823`；每个 job 的机器顺序为 RNG permutation，加工时间为 `[1,99]` 整数。
- validation/test 来自 OR-Library `jobshop1` 的 82 个经典实例。规范化后按 `(来源族、jobs、machines、content_hash)` 排序，偶数位置进入 validation、奇数位置进入 test，固定 41/41；所有 split 做 content-hash 去重。
- 保存原始来源、MIT 许可、下载 URL 和 SHA-256。来源说明：<https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html>；法律说明：<https://people.brunel.ac.uk/~mastjjb/jeb/orlib/legal.html>。
- population 冻结前，访问 guard 必须拒绝读取 test manifest。

### 4.2 JSSP `DomainAdapter`

- `JobShopInstance` 保存工序、机器数、来源和内容 hash。
- `JobShopSolution` 使用 operation-based job-ID sequence；确定性 schedule builder 按 job precedence 和机器最早可用时间解码，不依赖外部求解器。
- `ObjectiveEvaluation.scalar_cost = makespan`；分项包含 makespan、机器总空闲、关键路径长度，违规项包含 multiplicity 和未调度工序。
- guard 检查长度、job multiplicity、job 范围和有限性；codec 提供规范 JSON、clone 和稳定 hash。
- features 包含 critical-path ratio、瓶颈机器利用率、负载不平衡、critical block、operation displacement 和相对初始解改进率。
- trace encoder 使用与 UAV 相同的通用三态 recorder。

### 4.3 JSSP operator population 与 `jssp-v1` IR

- 固定八槽 P0：随机相邻交换、随机双点交换、有界插入、有界反转、critical-block 相邻交换、critical-block 端点交换、瓶颈 block 插入、高 idle-gap relocation。
- typed IR 将 selector、transform、repair 分开；所有参数有静态上下界，循环有硬上限。
- 变换保持 job multiplicity；异常、非法结果或超时统一安全 no-op。
- compiler 只能组合白名单 primitive；输入不可变、同 seed 确定性、有限时间和结构合法是硬门。
- sanity baseline 固定为随机 sequence、SPT dispatch 和 adjacent-swap hill climbing。

### 4.4 JSSP 完整流水线与验收

- smoke 使用 64 次搜索调用、2 代×2 候选；正式配置复用 UAV 的 400/240/400 调用和 3 代×3 候选×8 槽预算。
- validation 决定 retention；test 只比较冻结后的 P0/Pn。
- 报告 makespan、可行率、运行时间、有效调用率、接受率；有可靠 best-known 来源时额外报告 gap。
- 至少完成一个离线完整演化 smoke 和一个固定预算配对候选验证。
- core 源码不得导入 UAV 或 JSSP 类型；两个领域使用同一搜索、trace、diagnoser、memory、validation 和 candidate lifecycle。
- 创建 `cross-domain-core-qualification-v1` 标签。

## 5. 单仓三包与 core `0.1.0` 发布

状态：已完成。曾完成 history-preserving 独立仓库构建验证，但在任何新远端创建或推送前决定保留 monorepo；该尝试通过 `fc7ea29`/`893960a` 两个可审计提交完整记录且最终源码树未改变。core 独立构建验证提交为 `458cd12789096a61ea5276c7a7b1286fe3155828`，10 项契约测试及 wheel/sdist 通过；JSSP 独立构建验证提交为 `76dfbf1665edd98ef69573a4473dcb833545ba70`，30 项领域测试及 wheel/sdist 通过。这两个本地验证提交仅作为打包审计依据，不对应新 GitHub 仓库。monorepo 已增加 Windows/Linux、Python 3.11/3.12 CI、无源码复制的 core 构建器、API/迁移文档以及跨平台档案字节保护；整仓 wheel/sdist 和独立 core wheel/sdist 均构建成功，完整回归为 `343 passed, 3 skipped, 1 deselected`。标签 `trajectory-core-v0.1.0` 固定源码提交 `4de3d6d5a39105d52f365865a952168c41b7284c`；[GitHub prerelease](https://github.com/summerwind0131/UAV-Operator-Evolution/releases/tag/trajectory-core-v0.1.0) 已上传六个资产且未发布 PyPI。wheel SHA-256 为 `ada971c3…6752c`，sdist SHA-256 为 `224c81be…947b7`，release receipt 位于 `artifacts/releases/trajectory-core-v0.1.0.receipt.json`。

### 5.1 单仓包边界

- `src/operator_evolution_core`：通用协议、搜索、轨迹、诊断、memory、审计、提案信封、验证统计及 adapter contract kit。
- `src/uav_operator_evolution`：UAV adapter、IR/compiler、环境、benchmark、AFL-UAV 和研究 artifact。
- `src/jssp_operator_evolution`：JSSP adapter、IR/compiler、数据 manifest、baseline 和实验。

### 5.2 单仓发布流程

- monorepo CI 覆盖 Python 3.11/3.12、Windows/Linux、完整回归、core contract suite 以及 wheel/sdist build。
- 从同一不可变仓库提交提取 `operator_evolution_core` 构建独立分发包，源码不得复制成第二份长期维护版本。
- 在现有 `UAV-Operator-Evolution` 仓库发布 GitHub prerelease `trajectory-core-v0.1.0`，附 wheel、sdist、SHA-256、API 文档和迁移说明，不发布 PyPI。
- UAV/JSSP 在 monorepo 中直接引用同一提交内的 core；release receipt 同时记录仓库 tag、commit、提取规则和 wheel hash。

## 6. 第三阶段：UAV↔JSSP 双向机制迁移

### 6.1 机制协议与安全边界

- core 新增 `MechanismRecordV1`：来源领域、`repair/diversify/intensify/rollback` 标签、触发上下文、预期效果、失败模式、证据引用和 provenance hash。
- 跨领域只传递机制记录，不传 selector、primitive、IR、程序或领域特征原值；目标领域必须重新设计，并由自己的 compiler/validation gate 判定是否合法。

### 6.2 预注册实验设计

- 每个方向设置三臂：从零设计、同领域迁移、跨领域迁移。
- 每臂使用 10 个相同 master seeds、相同搜索/候选/Agent 预算；机制检索固定 top-4，依次按上下文相似度、证据强度、mechanism ID 决胜。
- 机制库只由独立 bank seeds 的 train/validation 运行建立，不含 test 结果。
- 默认使用确定性 heuristic/Mock Agent；真实远程 Provider 不属于完成门，未来加入时必须另行预注册。
- JSSP 最终比较使用已封存的 41 个 test instances。
- UAV 新建独立 `uav2d-transfer-v1`：120 张地图、六类各 20 张，并与所有旧 train/validation/test/Hidden Test-v2 按 content、terminal、obstacle-layout、geometry 和 seed 全部去重；population 冻结前保持 sealed。
- UAV 与 JSSP 最终评价比较 P0、从零、同领域、跨领域四组；每组使用相同 10-seed schedule。
- 使用实例级配对结果、95% bootstrap CI、Holm 多重比较校正；先比较可行率，再比较可行成本/makespan。
- 负迁移、零效果和失败上下文与正结果同等报告。完成不要求跨领域臂必胜，只要求实验有效、可复现且结论不夸大。
- 创建 `mechanism-transfer-v1` 标签和跨三仓复现包。

## 7. 持续测试、交付与默认边界

- Phase 1 每一步持续要求 UAV golden identity、旧 API、SQLite/JSONL、bundle/spec hash、审计顺序和 retention 决定不变。
- adapter contract suite 统一检查 clone 隔离、规范 JSON/hash、结构 guard、有限数值、异常 no-op、RNG 确定性和 test-split 隔离。
- 性能测试与确定性正确性测试分开；性能使用隔离进程、多次运行和中位数，不由单次 1 秒墙钟决定整套 CI。
- 所有研究运行保存 config、manifest、seed schedule、依赖、代码 commit、artifact hash、receipt、原始结果和分析版本。
- 不覆盖历史实验或 frozen artifact；不可避免的协议变更使用新 schema/version/addendum。
- 本计划不包含三维 UAV、动力学、移动障碍、真实飞控、动态 JSSP、机器故障、多目标 Pareto 或任意生成代码执行；这些只作为后续独立项目。

## 8. 实施日志

| 日期 | 里程碑 | 提交 / receipt | 测试与备注 |
| --- | --- | --- | --- |
| 2026-08-20 | Step 0–1 | `d44545f` | 领域协议与 UAV 纯适配完成 |
| 2026-08-20 | Step 2 | `eda6987` | 完整 UAV `DomainAdapter` 完成 |
| 2026-08-17 | Hidden Test-v2 执行 | execution receipt `d5ddb565…` | 6,960/6,960 唯一记录，0 API 调用 |
| 2026-08-17 | Hidden Test-v2 审计 | audit receipt `6789dc49…` | `passed`；冻结 v1 不再重跑或调参 |
| 2026-08-29 | Hidden Test-v2 收口 | `a8be8f6`，`uav2d-hidden-test-v2-final-v1` | 222 文件确定性归档，SHA-256 `06551f6d…cc30c`；Release 已发布；`291 passed, 3 skipped` |
| 2026-08-29 | Step 3 通用搜索内核 | `39a835b` + UAV routing commit | 逐步 operator/solution/evaluation/acceptance/temperature/stagnation 与最终 RNG shadow 一致；`295 passed, 3 skipped` |
| 2026-08-29 | Step 4 通用轨迹与诊断 | `refactor: separate core traces from UAV snapshots` | `instance_id`/`map_id` 无迁移兼容，UAV encoder snapshot 与旧 payload identity 一致；`298 passed, 3 skipped` |
| 2026-08-29 | Step 5 通用候选验证 | `refactor: generalize paired candidate validation` | CRN/ABBA/bootstrap/retention/slot replacement 已进入 core；UAV legacy 与 deterministic-v2 分离；默认 `302 passed, 3 skipped, 1 deselected`，性能 `1 passed` |
| 2026-08-29 | Step 6 提案信封与领域 IR | `refactor: separate proposal envelope from UAV IR` | `proposal-envelope-v1` 与 `UAVDomainKit` 完成；旧 bundle/proposal hash 冻结，错域/错版本/篡改 fail-closed；`307 passed, 3 skipped, 1 deselected` |
| 2026-08-29 | Step 7 演化管理器依赖注入 | `refactor: inject generic evolution dependencies` | population freeze 后才开放 test；UAV 全套 gate `310 passed, 3 skipped, 1 deselected`；中位性能变化 `-0.594%`，标签 `uav-generalization-phase1-v1` |
| 2026-08-30 | JSSP 领域基础与注册 smoke | `98e583e`–`3d24802`；receipt `697e5fd8…db38` | 60/41/41 content-disjoint split、八槽 `jssp-v1`、共享 search/trace/diagnosis/validation/lifecycle；64 calls、2×2 candidate smoke，1,024 traces，test 未打开 |
| 2026-08-30 | JSSP 正式 qualification | `89861f8`；receipt `ad44cc84…2f7d` | 400/240/400、3×3、8 槽；24,000 training traces；9/9 候选未保留；41/41 test P0/Pn 打平，可行率 1.0；零效果冻结 |
| 2026-08-30 | 单仓三包决策 | `fc7ea29` / `893960a` | 在新远端创建前取消三仓发布；core/UAV/JSSP 保留在现有仓库，同提交共同版本化 |
| 2026-08-30 | core `0.1.0` prerelease | `4de3d6d`；标签 `trajectory-core-v0.1.0` | 343 passed、3 skipped、1 deselected；独立 wheel/sdist 与六项 Release 资产发布；wheel SHA `ada971c3…6752c` |
