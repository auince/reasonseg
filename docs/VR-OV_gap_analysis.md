# VR-OV 实现差距分析

> 对比文档：`idea8_VR-OV_detailed_proposal.md` vs 当前仓库实际代码

---

## 总览

| 模块 | 状态 | 完成度 | 说明 |
|------|------|--------|------|
| 基础设施 (SAM2+BEiT-3) | ✅ 已有 | 100% | OpenWorldSAM base |
| 模块A (LLM查询解析器) | ⚠️ 部分 | 40% | BEiT-3编码+ParserHead有，GNN图推理无 |
| 模块B (场景图视觉编码) | ❌ 缺失 | 0% | 仅resources/有参考代码 |
| 模块C (组合式特征匹配) | ❌ 缺失 | 0% | 完全不存在 |
| 模块D (迭代式精化解码器) | ⚠️ 部分 | 10% | two_stage_inference有，3阶段精化无 |
| 训练策略 | ❌ 缺失 | 0% | 多任务loss+课程学习都不存在 |
| 数据生成(L1-L4) | ⚠️ 部分 | 15% | 仅L2空间级有mask生成器 |
| VR-OV-Bench | ❌ 缺失 | 0% | 完全不存在 |
| 本地数据集 | ⚠️ 部分 | 50% | 数据文件有，dataloader无 |

---

## 1. 基础设施 — ✅ 已有

| 组件 | 文件 | 状态 |
|------|------|------|
| SAM2 Hiera-L backbone | `reasonseg/modeling/open_world_sam2.py` | ✅ 已冻结 |
| BEiT-3 text encoder | `reasonseg/modeling/open_world_sam2.py:420-424` | ✅ 已冻结 |
| SAM2 prompt_encoder | `sam2/modeling/sam/prompt_encoder.py` | ✅ 已复用 |
| SAM2 mask_decoder | `sam2/modeling/sam/mask_decoder.py` | ✅ 已复用 |
| CrossAttentionTransformer | `reasonseg/modeling/open_world_sam2.py:470-477` | ✅ 已有 |
| two_stage_inference | `reasonseg/modeling/open_world_sam2.py` | ✅ 已有 |
| BEiT-3权重加载 | `checkpoints/beit3_large_patch16_224.pth` | ✅ 已下载 |
| SAM2权重 | `checkpoints/sam_vit_h_4b8939.pth` | ✅ 已下载 |
| RefCOCO pipeline | `scripts/train.py`, `eval.py`, `test.py` | ✅ 可用 |
| ReasonSeg config | `configs/refcoco/refcoco_base.yaml` | ✅ 可用 |
| LEARNED_PARSER config | `reasonseg/modeling/open_world_sam2_config.py:42-47` | ✅ 已定义 |

---

## 2. 模块A: LLM查询解析器 — ⚠️ 40%完成

| 子组件 | 提案路径 | 仓库状态 | 说明 |
|--------|----------|----------|------|
| LLM查询解析 (Step 1) | `model/query_parser.py` | ⚠️ 部分 | `reasonseg/query.py` 仅规则解析；LLM标注路径 `model/BIOtagging/llm_annotator.py` 存在但只做BIO标注 |
| BEiT-3文本编码 (Step 2) | BEiT-3 | ✅ 已有 | `open_world_sam2.py:420-424` |
| GNN图推理 (Step 3) | `model/gnn.py` | ❌ 缺失 | 完全不存在，resources/SGC-Net有GAT参考 |
| relation_embed (50种) | 模块A | ❌ 缺失 | 不存在可学习关系嵌入 |
| node_proj MLP (768→256) | 模块A | ⚠️ 部分 | `model/BIOtagging/query_parser_head.py` 的 `transformer` 可复用，但非专用 |
| QueryGraphGAT (2层GAT) | `model/gnn.py` | ❌ 缺失 | 不存在图神经网络 |
| `LLMQueryParser` 类 | `model/query_parser.py` | ❌ 缺失 | 不存在 |

**当前可用资产**：
- BEiT-3 模型和分词器已加载可用
- `model/BIOtagging/llm_annotator.py` 可调用DeepSeek API
- `model/BIOtagging/query_parser_head.py` 有TransformerEncoder可复用
- `model/BIOtagging/bio_schema.py` 有14类BIO标签定义

**需要新建**：
- `model/gnn.py`：QueryGraphGAT（2层GAT，4头注意力）
- `model/query_parser.py`：LLMQueryParser完整类
- 关系类型嵌入（50×768 ≈ 38.4K参数）

---

## 3. 模块B: 场景图增强视觉编码 — ❌ 0%完成

| 子组件 | 提案路径 | 仓库状态 | 说明 |
|--------|----------|----------|------|
| input_proj (3×Conv1x1) | 模块B | ❌ 缺失 | 核心代码不存在 |
| region_head (Top-K=64) | 模块B | ❌ 缺失 | 核心代码不存在 |
| hoi_tokens (5×256) | 模块B | ❌ 缺失 | 核心代码不存在 |
| hoi_cross_attn (MHA) | 模块B | ❌ 缺失 | 核心代码不存在 |
| sg_conv (场景图注入) | 模块B | ❌ 缺失 | 核心代码不存在 |
| gate (可学习门控) | 模块B | ❌ 缺失 | 核心代码不存在 |
| SceneGraphVisualEncoder类 | `model/scene_graph_encoder.py` | ❌ 缺失 | 文件不存在 |

**当前可用资产**：
- SAM2 backbone 已冻结可用
- `open_world_sam2.py` 有 `_bb_feat_sizes` 和多尺度特征提取流程可参考

**外部参考**（未集成）：
- `resources/code/SGC-Net/SGC-Net-main/models/model.py`：HOI Residual Attention + gate_weight 设计
- `resources/code/FleVRS/FleVRS-main/model/flexible_hoi.py`：HOI token 交互机制

**需要新建**：
- `model/scene_graph_encoder.py`：完整模块（~5.8M参数）

---

## 4. 模块C: 组合式特征匹配 — ❌ 0%完成

| 子组件 | 提案路径 | 仓库状态 | 说明 |
|--------|----------|----------|------|
| BCM (类别匹配) | 模块C | ❌ 缺失 | 点积相似度匹配不存在 |
| ATTM (属性匹配) | 模块C | ❌ 缺失 | 颜色/材质/尺寸投影不存在 |
| RSM (关系匹配) | 模块C | ❌ 缺失 | 场景图验证匹配不存在 |
| ACMM (动作匹配) | 模块C | ❌ 缺失 | HOI交叉注意力匹配不存在 |
| CMF (跨模态融合,3层Cross-Attn) | `model/cross_attention.py` | ❌ 缺失 | 专用CMF不存在 |
| CompositionalFeatureMatcher类 | `model/compositional_matcher.py` | ❌ 缺失 | 文件不存在 |
| color_proj/material_proj | 模块C | ❌ 缺失 | 属性感知投影不存在 |
| relation_matcher MLP | 模块C | ❌ 缺失 | 不存在 |
| action_matcher MHA | 模块C | ❌ 缺失 | 不存在 |
| fusion_proj | 模块C | ❌ 缺失 | 不存在 |
| 4维匹配分数图 | 模块C | ❌ 缺失 | M_cat/M_attr/M_rel/M_act不存在 |

**当前可用资产**：
- `open_world_sam2.py` 的 CrossAttentionTransformer 可参考（非完全符合CMF需求）
- `model/BIOtagging/query_parser_head.py` 的 TransformerEncoder 可复用部分

**外部参考**：
- `resources/code/Prompt-DINO`：早期融合 BiAttention（可参考但不可直接复用）

**需要新建**：
- `model/compositional_matcher.py`：完整模块（~3.7M参数）
- `model/cross_attention.py` 或扩展现有实现

---

## 5. 模块D: 迭代式精化解码器 — ⚠️ 10%完成

| 子组件 | 提案路径 | 仓库状态 | 说明 |
|--------|----------|----------|------|
| Stage 1 (Coarse Localization) | 模块D | ⚠️ 部分 | `open_world_sam2.py` 的两阶段推理有类似逻辑 |
| Stage 2 (Attribute Verification) | 模块D | ❌ 缺失 | 属性验证门控不存在 |
| Stage 3 (Relational Refinement) | 模块D | ❌ 缺失 | 关系精化不存在 |
| positional_tokens | 模块D | ✅ 已有 | `open_world_sam2.py:293` |
| attr_threshold (可学习) | 模块D | ❌ 缺失 | 不存在 |
| score_weights (4个) | 模块D | ❌ 缺失 | 不存在 |
| IterativeRefinementDecoder类 | `model/refinement_decoder.py` | ❌ 缺失 | 文件不存在 |

**当前可用资产**：
- SAM2 mask_decoder 和 prompt_encoder 已可用
- `open_world_sam2.py:495-530` 有两阶段推理流程可参考
- `self.positional_tokens` 已定义

**需要新建**：
- `model/refinement_decoder.py`：完整3阶段精化解码器（~25.6K参数）

---

## 6. 训练策略 — ❌ 0%完成

| 子组件 | 提案位置 | 仓库状态 | 说明 |
|--------|----------|----------|------|
| L_mask (BCE+Dice) | 损失函数 | ✅ 已有 | OpenWorldSAM已有 |
| L_attr (属性匹配损失) | 损失函数 | ❌ 缺失 | 不存在 |
| L_rel (关系匹配损失) | 损失函数 | ❌ 缺失 | 不存在 |
| L_act (动作匹配损失) | 损失函数 | ❌ 缺失 | 不存在 |
| L_compose (组合匹配损失) | 损失函数 | ❌ 缺失 | 不存在 |
| Phase 1 预训练流程 | `train_net.py` | ❌ 缺失 | 模块B/C单独预训练不存在 |
| Phase 2 端到端联合训练 | `train_net.py` | ❌ 缺失 | 不存在 |
| Phase 3 精调 | `train_net.py` | ❌ 缺失 | 不存在 |
| 课程学习 (L1→L4渐进) | 训练 | ❌ 缺失 | 不存在 |
| 查询图Dropout (p=0.2) | 训练 | ❌ 缺失 | 不存在 |
| EMA (权重指数移动平均) | 训练 | ❌ 缺失 | 不存在 |
| 属性级难负样本挖掘 | 训练 | ❌ 缺失 | 不存在 |

**当前可用资产**：
- `scripts/train.py` 和 `reasonseg/runtime/train.py` 有基础训练循环
- OpenWorldSAM的mask损失已实现

**需要新建**：
- 多任务联合损失模块
- 训练阶段管理逻辑
- 课程学习调度器

---

## 7. 数据生成 (L1-L4) — ⚠️ 15%完成

| 层级 | 提案数据规模 | 仓库状态 | 说明 |
|------|-------------|----------|------|
| L1 (属性级) | ~50K | ❌ 缺失 | COCO+Attributes→LLM生成不存在 |
| L2 (空间级) | ~40K | ⚠️ 部分 | `model/BIOtagging/seg_mask_generator.py` 有bbox空间关系→查询生成，但仅20K且无LLM增强 |
| L3 (动作级) | ~30K | ❌ 缺失 | HICO-DET/SWIG→LLM生成不存在 |
| L4 (嵌套级) | ~20K | ❌ 缺失 | 多数据集组合+LLM生成不存在 |
| 总计 | ~140K | ⚠️ | 仅约20K L2级数据（mask生成器） |

**当前可用资产**：
- `model/BIOtagging/seg_mask_generator.py`：空间关系查询生成
- `model/BIOtagging/llm_annotator.py`：DeepSeek API调用
- `scripts/synthesize_reviewed_mask_silver.py`：Flash+Pro审核流程
- `model/BIOtagging/data/expanded_silver_train50k_plus_mask20k.json`：已有69K数据（L2级为主）

**本地数据集**（文件存在，但无对应dataloader）：
| 数据集 | 路径 | 状态 |
|--------|------|------|
| COCO-Attributes | `dataset/coco_attributes/cocottributes_eccv_version.pkl` | ✅ 文件有 |
| Visual Genome Attributes | `dataset/vg_attributes/` | ⚠️ 目录存在 |
| HICO-DET | `dataset/hico_det/` (images+annotations) | ✅ 文件有 |
| SWIG-HOI | `dataset/swig_hoi/` (images+train/test json) | ✅ 文件有 |
| PACO-LVIS | `dataset/paco_lvis/paco_lvis_v1.zip` | ⚠️ 压缩包未解压 |

**需要新建**：
- L1属性级数据生成脚本（基于COCO Attributes + LLM）
- L3动作级数据生成脚本（基于HICO-DET/SWIG + LLM）
- L4嵌套级数据生成脚本（多数据集组合）
- 各数据集对应的 PyTorch Dataset 类

---

## 8. VR-OV-Bench 基准测试 — ❌ 0%完成

| 子组件 | 提案规模 | 仓库状态 |
|--------|---------|----------|
| VR-OV-Attr (属性级) | 3000样本 | ❌ 不存在 |
| VR-OV-Spatial (空间级) | 2000样本 | ❌ 不存在 |
| VR-OV-Action (动作级) | 2000样本 | ❌ 不存在 |
| VR-OV-Nested (嵌套级) | 1000样本 | ❌ 不存在 |
| VR-OV-ZeroShot (零样本) | 2000样本 | ❌ 不存在 |
| cIoU / Attr-Acc / Rel-Acc 指标 | 评估代码 | ❌ 不存在 |
| Comp-Score综合指标 | 评估代码 | ❌ 不存在 |
| `tools/build_vr_ov_bench.py` | 构建脚本 | ❌ 不存在 |

**当前可用资产**：
- `reasonseg/benchmarks/` 有基础benchmark框架
- `benchmarks/refexp_paper_benchmark.json` 有slice评估范本
- `scripts/benchmark/run_benchmark.py` 可参考

---

## 9. 项目文件结构 — 现有 vs 缺

### ✅ 已存在的文件

| 提案文件 | 仓库对应 | 说明 |
|----------|----------|------|
| model/sam2/ | `model/segment_anything_2/` | SAM2代码 |
| (未列) | `model/unilm/beit3/` | BEiT-3代码 |
| (未列) | `reasonseg/modeling/open_world_sam2.py` | 主模型 |
| (未列) | `reasonseg/modeling/open_world_sam2_config.py` | 配置定义 |
| (未列) | `scripts/train.py`, `eval.py`, `test.py` | 训练/评估/测试入口 |
| (未列) | `configs/refcoco/refcoco_base.yaml` | 基础配置 |
| (未列) | `model/BIOtagging/query_parser_head.py` | Parser Head（可视为模块A子集） |
| (未列) | `model/BIOtagging/bio_schema.py` | BIO标签定义 |

### ❌ 完全缺失的文件

| 提案文件 | 用途 |
|----------|------|
| `model/vr_ov.py` | VR-OV主模型 |
| `model/query_parser.py` | 模块A：LLM查询解析器 |
| `model/scene_graph_encoder.py` | 模块B：场景图增强视觉编码器 |
| `model/compositional_matcher.py` | 模块C：组合式特征匹配 |
| `model/refinement_decoder.py` | 模块D：迭代式精化解码器 |
| `model/gnn.py` | GNN图推理（模块A子组件） |
| `model/cross_attention.py` | CMF跨模态融合（模块C子组件） |
| `data/synthetic_query_gen.py` | 合成数据生成 |
| `data/vr_ov_dataset.py` | 数据加载 |
| `train_net.py` | 训练入口 |
| `demo.py` | 演示脚本 |
| `tools/build_vr_ov_bench.py` | Benchmark构建 |
| `tools/analyze_errors.py` | 错误分析 |
| `configs/vr_ov_base.yaml` | VR-OV基础配置 |
| `configs/vr_ov_full.yaml` | VR-OV完整配置 |

---

## 10. 外部参考代码现状

| 方法 | 本地路径 | 集成状态 |
|------|----------|----------|
| SGC-Net (HOI + gate) | `resources/code/SGC-Net/SGC-Net-main/` | ❌ 未集成 |
| FleVRS (HOI decoder) | `resources/code/FleVRS/FleVRS-main/` | ❌ 未集成 |
| Prompt-DINO (BiAttention) | `resources/code/Prompt-DINO/` | ❌ 未集成 |
| segllm (seg+LLM) | `resources/code/segllm-main/` | ❌ 未集成 |
| OVSAM (SAM2CLIP) | `resources/code/OVSAM/` | ⚠️ 目录存在但可能为空 |
| Semantic-SAM | 不在resources/ | ❌ 缺失 |
| COSINE | 不在resources/ | ❌ 缺失 |

---

## 11. 工作量估算（已实现 vs 待实现）

| 阶段 | 提案耗时 | 当前进度 | 剩余工作量 | 关键缺失 |
|------|---------|----------|-----------|----------|
| 基础框架 | 2周 | ✅ 完成 | — | — |
| 模块A (GNN部分) | 2周 | ⚠️ 40% | ~1.2周 | GNN图推理 |
| 模块B | 2周 | ❌ 0% | ~2周 | 场景图生成完整模块 |
| 模块C | 2周 | ❌ 0% | ~2周 | 四路匹配+CMF |
| 模块D (精化部分) | 1周 | ⚠️ 10% | ~0.9周 | Stage2/3精化逻辑 |
| 数据生成(L1-L4) | 2周 | ⚠️ 15% | ~1.7周 | L1/L3/L4生成 |
| 训练+消融 | 8周 | ❌ 0% | ~8周 | 全流程 |
| Benchmark | 4周 | ❌ 0% | ~4周 | VR-OV-Bench构建 |
| 论文 | 4周 | ❌ 0% | ~4周 | — |

---

## 12. 优先级排序（建议实施顺序）

| 优先级 | 组件 | 理由 |
|--------|------|------|
| P0 | 模块A：GNN图推理 | Parser Head已有，需补齐GNN即可获得完整模块A |
| P0 | 数据：L1属性级生成 | 本地COCO-Attributes已有，可最快扩大数据规模 |
| P1 | 模块C：组合特征匹配 | 核心创新，直接决定组合泛化能力 |
| P1 | 数据：L3动作级生成 | HICO-DET/SWIG-HOI本地数据已有 |
| P2 | 模块B：场景图视觉编码 | 5.8M参数最多，但价值最高 |
| P2 | 模块D：迭代精化解码器 | 依托现有two_stage_inference扩展 |
| P2 | 数据：L4嵌套级生成 | 需要前三个层级数据就绪 |
| P3 | VR-OV-Bench | 评估体系，训练完成后再建 |
| P3 | 训练策略+课程学习 | 各模块就绪后再实现 |

---

*分析日期：2026-04-30 | 基于 `idea8_VR-OV_detailed_proposal.md` v1.0*
