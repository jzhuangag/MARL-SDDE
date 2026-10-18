# V3 TSP定位、全文叙事与语言复核

日期：2026-10-18。复核基线为作者信息版a667d32；对象为独立修订目录的
main.tex、appendices.tex、37条参考文献、原有证据/技术审计，以及本轮编译的
全部13页。原始TSP-V3及所有科学数据、协议、门槛不变。

## 总体判断

**V3有合理的TSP投稿定位和实质内容；不能据此宣布创新性已获独立认证或达到稳录水准。**
最有说服力的主线是相关Markov观测下的有限预算学习/资源联合设计：资源动作
同时改变单步误差、可支付更新次数和辨识是否值得做。它应作为统计学习与策略
评估的资源控制论文来评判，而不是新型异步MARL优化器的全面性能论文。

Scope有充分依据，不需要编造无线系统或给MuJoCo套通信背景。TSP官方范围
包括信息处理、学习、估计和性能分析；其2025年SDDE/分布式SGD论文也是直接
相关的已发表先例。但“属于随机优化”不等于自动符合任何TSP选题：本文与
相关观测、时序估计和资源受限学习的联系才是具体依据。

官方作者说明同时强调实质新颖性，明确把已知理论/算法的直接组合列为可能
立即拒稿的情形。因此，本轮不以证明数量、参考文献数量或运行行数代替创新性。

## 六个维度

| 维度 | 评价 | 判断依据与实际风险 |
|---|---|---|
| Scope | 合适且有直接先例 | 分布式策略评估、相关Markov数据、Lyapunov性能分析、采样/通信资源设计；无需虚构应用 |
| 立意/意义 | 有价值 | 问题从“更多样本是否更准”推进到“预算结束时是否学得更好”，并把辨识成本计入剩余学习机会 |
| 创新性 | 中等、可争辩，仍是主要审稿风险 | 核心应是协方差/间隔/步长/可支付时域的耦合设计与learning-value admission；高斯有效样本量、Lyapunov收缩、KL链式法则本身不是新方法 |
| 叙事 | 原稿可读但偏审计化；本轮已改善 | 清理实验编号开头、重复数字、工程核验句和名词堆叠，以资源决策问题串起理论与实验 |
| 理论深度 | 有实质推导，不等于新的普适MARL理论 | 有限时域延迟界、正部滑动历史、条件critic跟踪、信息/机会下界、辨识准入与阈值；适用条件和证书相对保证必须保留 |
| 实验深度 | 对资源控制主张有分层支撑；对全新异步MARL算法领先仍不足 | 有机制、估计学习曲线、消融、64种子确认和两类合作环境迁移；核心MARL比较仍是固定参与数控制，而非所有强异步更新算法 |

建议继续TSP-first，不让V3等待140h实验；也不将该实验未来可能的正负结果
未经独立判断并入V3。若主张升级为“新异步MARL更新方向稳定优于强算法”，
那是独立科学任务，需要不同的直接证据，当前V3不能承担这一表述。

## 最近工作与创新边界

本轮有界检索使用公开网页搜索与IEEE/PMLR/SPS原始来源，日期2026-10-18。
关键词包括完整SDDE论文题名、DASA、delayed Markov stochastic approximation、
Lyapunov device scheduling、correlated temporal difference participation，以及
learning-aware finite-budget sampling。仅采用可追溯原始来源，不据搜索无结果
宣布“首篇”。下表主要依据原始摘要/介绍与既有项目context矩阵，不冒充完整
逐式比对所有先行论文。

| 已有工作 | 已经解决的相关问题 | V3应明确的差别 |
|---|---|---|
| Yu/Chen/Poor,TSP2025,10.1109/TSP.2025.3546574 | 用SDDE分析/优化异步SGD的worker activation与staleness | V3把跨actor协方差、物理采样间隔与有限可支付更新数放进终端风险；不能把“SDDE调度”本身当新颖点 |
| Perazzone等,2025,10.1109/TON.2025.3539857 | 已用收敛界与Lyapunov drift-plus-penalty做client/power调度 | 不能宣称首次由Lyapunov/收敛界设计资源；应展示本问题的相关数据与剩余学习价值如何改变动作 |
| Adibi等,AISTATS2024；DASA,CDC2024 | Markov采样下的延迟SA；DASA处理独立agent链与平均延迟/并行加速 | V3的有界确定性延迟证书不应被包装成比其延迟阶更强；差别在相关agent与预算化资源联合控制 |
| Gupta等,NeurIPS2019 | Lyapunov有限时域分析及自适应学习率 | V3还联动参与数、间隔与动作相关可学习时域；仅改变学习率不是充分差别 |
| Nitinawarat/Veeravalli2015及识别文献 | 受控Markov观测、非均匀信息成本、change-of-measure工具 | V3准入还比较识别后剩余学习风险；KL恒等式和occupation表达是工具，不能独立夸作普适新原理 |

## 理论复核结论与不能靠润色消除的问题

1. 已通读全部正式陈述和附录推导。本轮保留所有inline math、display math、
   assumption/theorem/proposition/corollary原文和有序引用命令，机器逐项比对通过。
   原有D/u/ell、scalar/dual预算和TD-pair混合时钟修复不被语言改写破坏。
2. 最有价值的理论不是单独的相关高斯均值公式，而是把收缩和噪声合成在可支付
   时域上，并明确“识别可行”与“识别值得”不同。消融给该区别提供实际证据。
3. moving-policy tracking依赖uniform mixed-history条件；固定正步长噪声底与
   phase theorem的1/s风险类不同。证书比较针对fallback上界，不能转成实际轨迹
   一定优于fallback。这三项是科学条件，不是应删掉的防御性套话。
4. controlled-belief opportunity下界使用声明的加性机会损失；阈值定理在分离
   Gaussian风险类及明确的gap/budget-shift假设下成立。不能从它推出任意神经
   MARL的最优学习复杂度或高概率无害性。
5. 经验均值上界与理论界的中位比20.77、最大228.98仍完整报告。这说明界是
   保守设计工具，不支持“紧常数”叙事。基础条件较强、校准成本和保守性如何
   影响实际动作，仍可能受到严肃审查；润色不能改变这一事实。

同一主模型完成本轮复核和编辑，不代表独立专家证明认证，也未做全库优先权
检索、全文查重或接受概率估计。

## 实验证据与允许的推论

- 对信息优先探测，风险下降35.24%/9.79%，分母已修正；仍保留绝对差的同步
  单侧下界0.1612/0.0577、77/96改善和预算/安全记录。
- 固定q=4由16个development种子选择，64个独立confirmation种子支持参数/
  估计回报AUC比0.9009/0.8868及各自区间。每个固定q也有自己的间隔/步长，
  因而是资源动作控制比较，不是人为不给baseline调步长。
- 终端参数风险相对myopic下降12.20%，相对correlation-only下降64.22%；没有
  改写成所有指标均优。robust shield保持原动作/经验风险不变，也原样保留。
- MPE2.319%是regime-seed先归一化后平均；MaMuJoCo8.341%是先构造每种子的
  equal-regime mixture再归一化，不能混称同一种估计量。各自最强单一固定动作
  仍是比较对象。MaMuJoCo相对分regime固定envelope的-0.542%/-0.997%未删除。
- 两个环境的MAPPO迁移和32次dependence-matched选择支持资源控制的实用性。
  它们没有证明全套异步更新机制全面领先HAPPO/MAPPO/DASA。重复记录行数或
  byte replay属于可复现性，不是独立种子数或新环境数。
- 当前独立140h结果没有读取或用于本文。此轮没有跑任何新native实验或改变
  旧结果/seed/gate。

## 逐节编辑与复核

| 部分 | 保留的作用 | 本轮处理/剩余判断 |
|---|---|---|
| 标题/摘要 | 定义资源控制而非通用新优化器 | 标题不改；201词摘要先解释冗余信息，再给方法、证书保证和实证；保留证书相对范围与全部摘要数值 |
| I Introduction | 建立问题、最近方法、差别与贡献 | 减少“also/instead/additionally”补丁式排列；贡献一先说终端资源设计，普通Gaussian事实作辅助；去掉与摘要/结果重复的一组数字 |
| II Model | 固定策略、混合、独立延迟口径 | 全部公式/假设原文保留；Gaussian与有界TD两类模型的角色明确，D/u/ell区别保留 |
| III Geometry | 说明相关性/间隔如何改变资源效率 | 清晰推导原样保留，没有为“去AI味”机械换词或篡改已知统计结论 |
| IV Lyapunov control | 延迟界、条件跟踪、终端score | 保留全部条件与零延迟边界；将单步比较小节改为中性题名；明确solver需要global accuracy证书 |
| V Adaptation | 受控信息、learning-value准入和phase | 实际冻结动作的upper certificate写法更直接；1/s风险类与正步长底区别仅在相关位置保留 |
| VI Implementation | 可实施选择及校准接口 | “不是plugin真值/不是乐观外推”改为肯定式有效上界/排除规则；coverage失败上的loss control仍保留 |
| VII Simulations | 机制→估计→迁移→消融的证据链 | 多个实验编号开头改成研究问题；去掉重复工程行数声明；不删负向/零结果、成本口径、统计区间或baseline定义 |
| VIII Conclusion | 收束已经支持的贡献 | 压缩为方法、辨识价值和证据三句，避免重新罗列所有理论标签和泛化承诺 |
| A–F Appendices | 完整证明 | 仅三处解释句去掉反驳式修辞；所有数学和正式陈述保持不变 |
| References/figures/author block | 科学和作者出处 | Bib、所有figure资产、作者块逐字节保持；37条fresh核验及raw限流/representation记录另存 |

## 修订守恒与交付状态

原始文件a667d32可从Git完整恢复。此轮采用LaTeX原生apply_patch，不将Markdown
HTML标记插入TeX；tsp_narrative_revision_check_20261018.json记录每个修改hunk的
前后文本、哈希、review-item映射和不变区域检查。这是原生适配记录，不冒充
执行过ARS的Markdown anchor/apply脚本。

机器门通过：所有数学序列/正式陈述/引用和交叉引用顺序不变；原始V3的冻结
哈希不变，figure/Bib/作者块不变，关键统计含不利结果均在。最终PDF13页、
201词无缩写摘要/五关键词、零undefined/overfull，24个underfull为窄栏排版警告。
全部13页重新渲染并逐页检查，字体嵌入；图表未裁切，无字体/边距压缩。
非实验内容约10.5页当量，符合9–11页目标；含proof/reference的初投总页数13。

PDF SHA256:289c065ee2d5fc440668724b4751d7d9ae723833caa2c3b1da56ab21eb9de431。
后续以本轮revision check为最新稿件指纹，不重跑历史author-only不变范围门。

Fresh citation原始结果为35 PASS：Paulin的article79/issue表示分歧另有新鲜机构
记录；Perazzone的本次Crossref遇429，保留不可用状态，以该429之前本轮取得的
IEEE原始publisher页面核对存在与出版信息，并交叉保留稍早同日fresh Crossref
作者元数据。没有重试/绕过限流或把raw状态改成PASS。两项分别写入resolution
sidecar；37个authoritative existence matches和服务可用性不是同一指标。
本轮引文内容/有序分组/Bib不变；新鲜核验后的局部压缩只改变叙述，最终逐处
context复核。IEEE style无明确错误；Paulin打印no.79是已记录的表示判断。

内部编辑性量表仅为诊断，不是TSP评审分数或接受概率：Originality6.5/10、
Rigor7/10、Evidence6.5/10、Coherence8/10、Writing8/10；按20/25/25/15/15权重
为7.075/10。写作改善不改变原创性/实验证据本身的质量。

结论：**TSP_CANDIDATE_EDITORIAL_REVIEW_NOT_ACCEPTANCE_CLEARANCE**。
真实作者ORCID、既往/并行/相关稿件披露、全作者同意/最终release仍待作者提供。
IEEE官方说明要求全作者ORCID；不再把它当成可由系统猜填的可选信息。不执行投稿。

## 原始网页来源

- Scope: https://signalprocessingsociety.org/publications-resources/ieee-transactions-signal-processing
- Novelty/format/author rules: https://signalprocessingsociety.org/publications-resources/information-authors
- SDDE: https://ieeexplore.ieee.org/document/10909566/
- Lyapunov scheduling: https://ieeexplore.ieee.org/document/10885536/
- Delayed Markov SA: https://proceedings.mlr.press/v238/adibi24a.html
- DASA: https://ieeexplore.ieee.org/document/10886380/
- Adaptive rates: https://papers.nips.cc/paper/2019/hash/e354fd90b2d5c777bfec87a352a18976-Abstract.html
- Controlled sensing author preprint: https://arxiv.org/abs/1310.1844
- Paulin identity: https://www.research.ed.ac.uk/en/publications/concentration-inequalities-for-markov-chains-by-marton-couplings-/

使用技能：academic-research-suite（revision与citation-integrity内联角色）、
ieee-tsp-paper-writing、publishable-academic-writing、kimi-academic-polish、
ieee-bib-style-audit、pdf:pdf。Kimi规则用于清晰和科学忠实，不保证检测器结果。
