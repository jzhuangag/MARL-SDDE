# Copy-paste prompt for the HPC4-connected Codex Agent

```text
你现在接手 MARL-SDDE 项目的 HPC4/GPU 阶段。请自主推进，不要为了普通实现细节反复询问我；但不得扩大研究 claim、改写失败结论、泄露凭证或执行未经授权的清理。

第一原则：在任何 HPC4 read、transfer、run、monitor、cleanup 之前，完整读取并严格使用你本机的 $hpc4 skill。学术实验规划、统计解释和复现使用 $academic-research-suite 的 experiment workflow。若涉及论文文字或引用，再按相应 academic skill 和 citation gate 处理。

项目与来源：
- GitHub: https://github.com/jzhuangag/MARL-SDDE.git
- branch: codex/joint-ms-exp007c
- 必须包含的 ancestor: c01b900
- 原电脑 canonical path: E:\HKUST-study\vin\SDDE
- HPC4 active root: /scratch/jzhuangag/MARL-SDDE
- HPC4 durable root: /project/vincentlau/jzhuangag/MARL-SDDE
- 首先完整阅读 docs/HANDOFF_HPC4_GPU.md，然后阅读其中列出的 proof、EXP-013B preregistration/validation 和 summary。

源代码硬门槛：
1. clone/fetch 后运行：
   git merge-base --is-ancestor c01b900 HEAD
2. 若返回非零，立即停止研究执行并告诉我：GitHub 仍缺少原电脑未推送的 5 个 commits，需要我提供已更新 branch 或 Git bundle。禁止根据交接摘要重建缺失代码或伪造结果。
3. 若通过，记录完整 HEAD、git status、remote 和 diff。保留用户已有修改。

HPC4 安全预检：
1. Windows 必须显式使用 C:\Program Files\OpenSSH-Win64\ssh.exe 和 alias hpc4；不要使用已损坏的 System32 OpenSSH。
2. 先检查 DNS/TCP 22/HKUST VPN，再做 key-only BatchMode 登录。不要打印、保存或自动填写密码、2FA、private key。
3. 登录后只读检查 id/hostname/uptime、quota、df、bounded du、squeue、sinfo。
4. 不要把环境、dataset、cache、checkpoint 或 run archive 放 /home/jzhuangag（home 只有 200 GB）。
5. active run 放 /scratch/jzhuangag/MARL-SDDE；完成的重要结果放 /project/vincentlau/jzhuangag/MARL-SDDE。HF_HOME、HF_DATASETS_CACHE、TORCH_HOME 指向 /project。
6. 根据 sinfo 选择真实存在的 GPU partition/resource，不得猜 partition 名。CUDA job unset ROCR_VISIBLE_DEVICES 和 HIP_VISIBLE_DEVICES。
7. 不使用 destructive mirror，不 broad delete；未得到明确清理授权前不要删除 scratch。

研究事实，必须原样保留：
- 当前主线建议题目：Beyond Linear Speedup: Safe Adaptive Participation under Correlated and Delayed Markov Data。
- 已证明 exact/minimax speedup q/[1+(q-1)rho]、predictably decorrelated delayed affine Markov-TD finite-time bound、mixing/correlation anytime certificates，以及固定参数的任意非线性更新 covariance identity [rho+(1-rho)/q]Sigma。
- Theorem 9 不是 global neural-TD convergence theorem。
- SDDE 当前是 Lyapunov–Krasovskii interpretation；SDDE-to-discrete approximation error 仍 open。
- EXP-013B 正式结论必须写成 failed 3/5 gates。它强力支持 nonlinear loss of parallel speedup，但没有证明 correlation-only 固定 q 稳定优于 all-agent。
- 禁止通过放宽 gate、删 seed、重跑挑 seed、post-hoc 改 endpoint 来“修复”EXP-013B。

你的具体任务按阶段执行：

Phase 0 — provenance/environment
- 验证 commit 和文档。
- 完成 HPC4 preflight、storage audit、GPU/Slurm capability audit。
- 在 home 外建立 project-pinned environment，记录 Python/PyTorch/CUDA/driver/GPU/package lock。
- 修改代码前运行完整 pytest。
- 建立 docs/hpc4_run_ledger.md，记录每个 job 的 commit/diff、命令、job ID、资源、日志、输出、验证和归档路径。

Phase 1 — state/risk-aware controller
- 不要继续调一个 correlation-only 固定 q。
- 实现 predictable、low-complexity 的 blockwise (q,b,eta) controller。
- risk surrogate 至少包含：
  (a) 当前 transient/progress state；
  (b) certified q_eff=q/[1+(q-1)rho_upper] variance term；
  (c) delay/stability constraint；
  (d) message 或 wall-clock cost；
  (e) uncertainty/tail-risk penalty。
- 下一 block 的决定只能使用此前可观测信息。
- 不使用 Hessian inverse、full covariance matrix 或 preconditioner；目标复杂度 O(qd) time、O(d) memory。
- baselines 至少包括 all-agent adaptive eta、fixed small q、correlation-only、delay-only、state/risk-aware 和 charged information oracle。
- 先用 implementation-only pilot seeds，记录完整 trajectory、q/b/eta、q_eff、messages、wall time、mean 和 CVaR/tail metric。pilot seeds 禁止进入 formal confirmation。

Phase 2 — formal controlled GPU confirmation
- pilot 稳定且问题可识别后，先写 docs/experiment_014a_*.md 并 commit。
- 冻结 seeds、budgets、controller hyperparameters、baselines、primary endpoints、cluster bootstrap、multiplicity 和全部 gates；记录 registration commit hash 后才运行。
- 使用全新 paired seeds，通过 Slurm 提交。queued 不是 failure；用 squeue/sacct/bounded tail 监控，不 duplicate submit。
- 无论结果 pass/fail，都生成 docs/validation_exp014a.md 和 machine-readable metrics/summary。
- primary question：state/risk-aware 是否在相同 resource budget 下同时降低 error 和 upper-tail risk，并可靠优于 all-agent 及 correlation-only，而不是只赢一个弱 baseline。

Phase 3 — standard multi-agent Markov benchmark
- 不要求 actor–critic。优先 shared neural TD、Q-learning、value evaluation 或能明确导出 per-agent stochastic update 的方法。
- 先检查 HPC4 上可维护、可安装的 benchmark；候选可包括 PettingZoo MPE/SISL。任何替换必须记录原因，不能静默换任务。
- 尽量至少 3 个 task/regime；formal comparison 使用相同 environment steps、message budget、evaluation episodes 和 paired seeds。
- 报告 task return/score、Bellman/value error（若有 reference）、wall time、environment steps、messages/bytes、divergence rate、q/b/eta、estimated/empirical q_eff、mean 和 tail risk。
- ablation 至少拆除 state、correlation certificate、mixing control、delay control、tail penalty。
- natural correlation 和 injected common-factor correlation 分开报告；后者只能称 causal stress test。

Phase 4 — validation/archive
- 完整 pytest。
- deterministic analysis 必须 hash/byte comparison；stochastic training 按 preregistered distribution tolerance，不比较 wall-clock equality。
- 保存 Slurm scripts、job IDs、logs、metrics、figures、必要 checkpoints 和 environment lock。
- 完成结果复制到 /project/vincentlau/jzhuangag/MARL-SDDE，并验证 count/hash。
- 未经我授权不要删除 scratch。

自主决策规则：
- 普通依赖安装、代码实现、测试、合理资源申请、sbatch 和监控都在本任务授权范围内，可以自主完成。
- crash 后不要 silent retry；先查 exit code、sacct 和日志，记录根因，再决定是否需要我授权改变 formal protocol。
- formal gate failed 时必须诚实保留 failure，并据此收窄 claim。
- 如果 standard benchmark 使“参与 agent”的含义与 theorem 不一致，停止并报告，不要用名称相似掩盖概念错位。
- 不要花 GPU 时间证明 Poisson equation 或 SDDE approximation；那些是理论任务。GPU 优先用于 state/risk controller 和标准 benchmark。

进度沟通：
- 开始时先回复：source ancestor 是否通过、HPC4 connectivity/preflight、计划使用的真实 partition/GPU、active/durable path。
- 每个 Slurm job 提交后报告 job ID、exact command、resources、logs、expected outputs。
- 长任务持续监控，不因 unchanged queue state 误判 failure。
- 遇到真实 blocker 才询问我；否则自动推进到 validation 和 archive。

最终交付：
1. provenance/HPC4 preflight；
2. state/risk-aware method、复杂度和 tests；
3. EXP-014A preregistration、jobs、formal honest pass/fail；
4. 至少 standard benchmark smoke，或准确的 dependency/data blocker；
5. run ledger、artifact absolute paths 和 durable archive verification；
6. ICML 2027 是否继续、需要补什么的审稿级判断；
7. git commit/push/PR 的真实状态，不得声称未验证的远端更新。
```
