# RecursiveMAS 技术报告

## 整体架构与 MAS 实现原理

RecursiveMAS 实现了一个基于**潜在空间通信（Latent Space Communication）**的多智能体系统。其核心创新在于：多个异构小语言模型通过**连续潜在向量**而非自然语言文本进行通信协作，仅由最终执行者在输出层生成可读文本。

### 文件分工

| 文件 | 职责 |
|---|---|
| `modeling.py` | 定义两类核心神经网络模块：**内部适配器**（inner adapter，单模型潜在推演）和**跨模型适配器**（outer adapter，模型间潜在映射） |
| `prompts.py` | 定义所有 MAS 拓扑的提示模板，关键设计是**槽位标记**（slot markers），用于在提示中嵌入潜在向量 |
| `run.py` | 入口编排器：解析 CLI 参数，解析模型/适配器路径，按 style→family 分发到对应推理模块 |
| `inference_utils/inference_mas.py` | **链式 MAS** 核心引擎（Planner→Refiner→Solver），包含潜在推演和递归反馈的完整实现 |
| `inference_utils/inference_mas_mixture.py` | **层次混合式 MAS**：多领域专家并行处理 + 汇总器整合 |
| `inference_utils/inference_mas_distill.py` | **教师-学生蒸馏式 MAS**：专家生成计划（潜在），学习者执行 |
| `inference_utils/inference_mas_deliberation.py` | **审议式 MAS**：反思者+工具调用者交替迭代，支持 Python 执行和网页搜索工具 |

---

## 1. 模型层：`modeling.py`

### 1.1 Adapter（内部适配器）

本质是一个带残差的 bottleneck MLP，用于单个模型内部进行潜在空间推演：

```
Adapter(hidden_size):
  pre_ln  → proj1(h→h) → GELU → proj2(h→h) → residual(+) → post_ln
```

- `pre_ln`: 输入归一化
- `proj1`: 线性投影（保持维度不变，hidden_size → hidden_size）
- `GELU`: 非线性激活
- `proj2`: 线性投影
- 残差连接：`out = x + proj2(GELU(proj1(pre_ln(x))))`
- `post_ln`: 输出归一化

**用途**：在潜在推演（latent rollout）的每一步中处理当前最后隐藏状态，产生下一步的嵌入向量，实现模型在潜在空间中的"思考"。

### 1.2 CrossModelAdapter（跨模型适配器）

处理不同模型之间隐藏维度不同的映射问题：

```
CrossModelAdapter(in_dim, out_dim):
  ln_source → proj1(in→2*out) → GELU → proj2(2*out→out) → residual(Lin(in→out)) → ln_target
```

- 中间隐藏维度为 `out_dim * 2`（扩展瓶颈）
- `residual_proj`: 线性投影 `in_dim → out_dim` 解决维度不匹配
- `ln_source` / `ln_target`: 源/目标空间归一化

**用途**：将 Agent A 的潜在向量映射到 Agent B 的嵌入空间。例如 Planner 的输出是 896 维（1.7B 模型），需要映射到 Refiner 的 1024 维（1B 模型）空间。

---

## 2. 提示层：`prompts.py`

### 2.1 槽位机制

系统通过特殊槽位标记实现**潜在向量注入**：

```python
# 链式 MAS
PLANNER_SLOT = "<<LATENT_PLANNER_SLOT>>"       # 注入 planner 潜在向量
REFINED_SLOT  = "<<LATENT_REFINED_SLOT>>"       # 注入 refiner 潜在向量
FEEDBACK_SLOT = "<<LATENT_FEEDBACK_SLOT>>"       # 注入 solver 反馈潜在向量

# 层次式 MAS
HIE_MATH_EXPERT_SLOT    = "<<HIE_MATH_EXPERT_SLOT>>"
HIE_CODE_EXPERT_SLOT    = "<<HIE_CODE_EXPERT_SLOT>>"
HIE_SCIENCE_EXPERT_SLOT = "<<HIE_SCIENCE_EXPERT_SLOT>>"

# 蒸馏式 MAS
DISTILL_EXPERT_SLOT    = "<<DISTILL_EXPERT_SLOT>>"
DISTILL_FEEDBACK_SLOT  = "<<DISTILL_FEEDBACK_SLOT>>"

# 审议式 MAS
DELIBERATION_REFLECTOR_SLOT = "<<DELIBERATION_REFLECTOR_SLOT>>"
DELIBERATION_FEEDBACK_SLOT  = "<<DELIBERATION_FEEDBACK_SLOT>>"
```

### 2.2 两类提示函数

对于每种提示，提供两种构建方式：

- **`build_*_prompt_with_slot()`**: 保留槽位标记，用于潜在模式——实际运行时将槽位替换为前驱 Agent 的潜在向量（token embedding 形式）
- **`build_*_prompt()`**: 填充具体文本，用于纯文本基线模式

### 2.3 四种 MAS 拓扑的 Agent 角色

| 拓扑（family） | Agent 角色 | 信息流向 |
|---|---|---|
| **sequential**（链式） | Planner → Critic → Solver | 串行，每步将上一阶段输出嵌入提示 |
| **mixture**（层次混合） | Math Expert + Code Expert + Science Expert → Summarizer | 三专家并行提供信号，汇总器合成最终答案 |
| **distillation**（蒸馏） | Expert（教师） → Learner（学生） | 专家产生执行计划，学习者按计划生成答案 |
| **deliberation**（审议） | Reflector ⇄ Toolcaller | 反思者分析问题+使用工具，工具调用者综合信息生成答案 |

---

## 3. 入口层：`run.py`

### 3.1 主流程

```
CLI 参数解析 → 数据集加载 → 模型/适配器路径解析 → 投递到对应推理模块
```

关键参数：
- `--style`: 选择预定义模型组合（如 `sequential_light`）
- `--dataset`: math500 / medqa / gpqa / mbppplus
- `--num_recursive_rounds`: 递归轮数
- `--latent_steps`: 潜在推演步数
- `--batch_size`: 批大小

### 3.2 路径解析（`resolve_style_paths`）

根据不同 family 构建不同的模型和适配器路径字典：

- **sequential**: 3 个 agent 模型路径 + 3 个 inner adapter 路径 + 3 个 outer adapter 路径（outer_12, outer_23, outer_31）
- **mixture**: 4 个 agent 模型 + 6 个 outer adapter（expert→summarizer ×3, summarizer→expert ×3）
- **distillation**: 2 个 agent 模型 + 2 个 outer adapter（expert→learner, learner→expert）
- **deliberation**: 2 个 agent 模型 + 2 个 outer adapter（reflector→toolcaller, toolcaller→reflector）

### 3.3 潜在步数扫描

```python
LATENT_STEPS_SWEEP = (16, 32, 48)
```

默认对三个潜在步数进行扫描，选择准确率最优的结果作为最终输出。

---

## 4. 核心引擎：`inference_mas.py`（链式 MAS）

### 4.1 三种推理方法

`main()` 函数通过 `args.method` 支持三种模式：

| 方法 | 通信方式 | 递归 |
|---|---|---|
| `text` | 文本提示拼接 | 否 |
| `text_recursive` | 文本，上一轮 solver 输出作为下轮 planner 的反馈 | 是 |
| `ours` | 潜在向量（latent embeddings） | 否（单轮） |
| `ours_recursive` | 潜在向量 + solver→planner 反馈 | 是 |

### 4.2 潜在空间推演：`autoregressive_latent_rollout`

这是整个系统的核心机制，实现模型在潜在空间中"无文本思考"：

```python
for step in range(latent_steps):
    outputs = model(inputs_embeds, attention_mask, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][:, -1, :]   # 提取最后位置的隐藏状态
    next_embed = inner_adapter(last_hidden)               # 经 adapter 处理
    input_embeds = concat(input_embeds, next_embed)       # 拼接到嵌入序列末尾
    attention_mask = concat(attention_mask, 1)            # 扩展注意力掩码
return all_last_hidden_states
```

**关键设计**：
- 每一步只在最后一个 token 位置获取隐藏状态（通过 `logits_to_keep=1` 节省显存）
- `inner_adapter` 将隐藏状态转化为下一个"虚拟 token"的嵌入
- 这些连续的潜在向量形成了一条"思维链"，但全程没有文本 token 参与
- 最终返回所有步骤的隐藏状态序列

### 4.3 单轮潜在链式 MAS（`method == "ours"`）

```
                     latent rollout         latent rollout         文本生成
Planner模型 ──────────→ [P→R 潜在] ────→ Refiner模型 ──→ [R→S 潜在] ────→ Solver模型 → 最终答案
                   ↗ inner_adapter_1                   ↗ inner_adapter_2
                   ↘ outer_adapter_12                  ↘ outer_adapter_23
```

**阶段 1：`run_planner_latent_stage`**
1. 加载 Planner 模型和 tokenizer
2. 构建数学/代码规划提示
3. 将提示渲染为 token IDs
4. 执行 `autoregressive_latent_rollout(latent_steps)` 产生潜在推演序列
5. 潜在序列经过 `inner_adapter_1`（自处理），再经过 `outer_adapter_12`（维度映射），输出给 Refiner 的潜在向量

**阶段 2：`run_refiner_latent_stage`**
1. 加载 Refiner 模型
2. 构建带有 `<<LATENT_PLANNER_SLOT>>` 标记的提示
3. `split_prompt_ids_by_slots`：将提示切分为 `[前缀文本] + [SLOT] + [后缀文本]`
4. 前缀文本 → 嵌入，SLOT → Planner 的潜在向量（经维度对齐），后缀文本 → 嵌入
5. 拼接为混合嵌入序列（文本+潜在混合）
6. 执行潜在推演 → inner_adapter_2 → outer_adapter_23 → 输出给 Solver

**阶段 3：`run_solver_latent_stage`**
1. 加载 Solver 模型
2. 类似阶段 2，但注入的是 Refiner 的潜在向量
3. **直接调用 `model.generate(inputs_embeds=...)` 进行文本自回归生成**
4. 产生最终的自然语言答案

### 4.4 递归潜在 MAS（`method == "ours_recursive"`）

核心思想是构建**闭环反馈系统**：

```
Round 1: Planner → Refiner → Solver ──→ [feedback via outer_31] ──→ Planner (Round 2)
Round 2: [via outer_31] → Planner → Refiner → Solver ──→ [feedback] → ...
...
Round n: Planner → Refiner → Solver → 最终答案
```

**新增阶段：`run_solver_feedback_latent_stage`**
- Solver 收到 Refiner 潜在信号后，不生成文本，而是执行潜在推演
- 推演结果经过 `inner_adapter_3 → outer_adapter_31` 映射回 Planner 的嵌入空间
- 这个反馈向量代表 Solver 的"潜在状态信号"，包含对问题理解的修正信息

**新增阶段：`run_planner_feedback_latent_stage`**
- Planner 在后续轮次中，提示中包含 `FEEDBACK_SLOT` 槽位
- 将上一轮 Solver 的反馈潜在向量注入该槽位
- Planner 基于反馈重新执行潜在推演，产生改进后的计划

**递归终止**：最后一轮 Solver 执行文本生成（而非反馈生成），输出最终答案。

### 4.5 槽位嵌入注入的技术细节

`split_prompt_ids_by_slots` 的实现：

1. 将带槽位的用户提示渲染为完整的 chat template 文本
2. 在渲染后的文本中定位槽位标记的位置
3. 将文本切分为 `[prefix_text, slot_content, suffix_text]`
4. prefix 和 suffix 通过 tokenizer+embed_layer 转为嵌入
5. 槽位位置**直接插入来自其他 Agent 的潜在向量**（已通过 outer adapter 对齐维度）
6. 三部分拼接形成完整的输入嵌入序列

这种设计的优雅之处在于：LLM 的 transformer 天然能处理混合序列（文本嵌入 + 潜在向量），无需修改模型架构。

---

## 5. 其他 MAS 拓扑

### 5.1 层次混合式（Mixture / HIE）

```
                  ┌─ math_expert    ──→ outer_1s ──┐
Question text ────┼─ code_expert    ──→ outer_2s ──┼──→ Summarizer → 答案
                  └─ science_expert ──→ outer_3s ──┘
                         ↑                │
                         └── outer_s1/s2/s3 ←──┘  (递归反馈)
```

- 三个领域专家模型**并行**处理同一问题
- 各自的潜在推演结果通过独立的 outer adapter（1s, 2s, 3s）映射到 Summarizer 空间
- Summarizer 接收三个专家信号后生成文本答案
- 递归时：Summarizer 产生三个独立的反馈潜在向量（通过 outer_s1, outer_s2, outer_s3），分别回传给对应专家

### 5.2 蒸馏式（Distillation）

```
Expert ──→ outer_el ──→ Learner → 答案
   ↑                       │
   └── outer_le ←──────────┘ (递归反馈)
```

- Expert 接收问题，在潜在空间中生成"执行计划"
- Learner 将 Expert 计划作为注入的潜在信号，据此生成文本答案
- 递归时：Learner 的潜在状态（outer_le）反馈给 Expert，Expert 据此修订计划

### 5.3 审议式（Deliberation）

```
Reflector ──→ outer_rt ──→ Toolcaller → 答案
    ↑                          │
    └─── outer_tr ←────────────┘ (递归反馈)

工具: <python>...</python>, <search>...</search>
```

- Reflector 可以调用工具（Python 代码执行、Tavily 网页搜索）
- 通过自定义 `ToolTagStoppingCriteria` 实现工具标签触发停止
- Toolcaller 综合 Reflector 的分析结果生成最终答案
- 递归反馈通过 outer_tr 传递

---

## 6. 系统设计总结

### 6.1 核心创新点

1. **潜在空间通信**：Agent 间不通过文本交流，而是传递连续潜在向量。这避免了文本瓶颈（信息压缩损失）和文本生成的推理开销。

2. **异构模型协作**：通过 `CrossModelAdapter`，可以使用不同架构/尺寸的模型作为不同 Agent，每个模型的隐藏维度可以不同，由 adapter 统一对齐。

3. **潜在思维链**：`autoregressive_latent_rollout` 机制让模型在嵌入空间中执行多步推理，类似于链式思考（Chain-of-Thought），但不需要生成 token。

4. **递归自我修正**：通过反馈 outer adapter（outer_31, outer_le, outer_tr 等），系统形成闭环，后续轮次可以基于前一轮的"潜在状态信号"改进输出。

### 6.2 适配器生态

每种 MAS 拓扑需要的适配器组合：

| 拓扑 | Inner Adapter (×N) | Outer Adapter (交叉映射) |
|---|---|---|
| 链式    | 3 (planner, refiner, solver) | 3 (P→R, R→S, S→P_fb) |
| 层次混合 | 4 (3 expert + summarizer) | 6 (3→S, S→3) |
| 蒸馏    | 2 (expert, learner)         | 2 (E→L, L→E) |
| 审议    | 2 (reflector, toolcaller)   | 2 (R→T, T→R) |

### 6.3 推理管线总览

```
加载数据集 ──→ Pl 潜在推演 ──→ 潜在向量 ──→ Re 潜在推演 ──→ 潜在向量 ──→ So 文本生成 ──→ 答案
                                    │                                              │
                                    └──── [递归反馈: So→outer_31→Pl] ←────────────┘
```

### 6.4 支持的预训练模型组合

系统通过 `STYLE_SPECS` 预定义了多组已发布的 HuggingFace checkpoint，每组包含特定模型的适配器权重：
- **Sequential-Light**: Qwen3-1.7B + Llama3.2-1B + Qwen2.5-Math-1.5B（总参数量 ~4.2B）
- **Sequential-Scaled**: Gemma3-4B + Llama3.2-3B + Qwen3.5-4B（总参数量 ~11B）
- **Mixture**: DeepSeek-R1-Distill-Qwen-1.5B + Qwen2.5-Coder-3B + BioMistral-7B + Qwen3.5-2B
- **Distillation**: Qwen3.5-9B (expert) + Qwen3.5-4B (learner)
- **Deliberation**: Qwen3.5-4B ×2
