# VR-OV: Visual-Reasoning Enhanced Open-Vocabulary Unified Segmentation Framework

## 详细技术方案报告

---


## 1. 问题形式化定义

### 1.1 核心问题

给定输入图像和自然语言查询，现有开放词汇统一分割框架（如 OpenWorldSAM、COSINE、Prompt-DINO）仅能处理简单类别查询如 "car"/"person"。真实需求：

- "the woman in red dress who is watering flowers"
- "the black dog running behind the bicycle"
- "the half-eaten apple on the wooden table"

这些需要同时理解：**对象类别 + 属性 + 动作 + 空间关系 + 状态**。

### 1.2 形式化定义

**定义 1 (复杂组合查询 CCQ)**：一个复杂组合查询 Q 可被解析为一个结构化查询图 G_Q = (V_Q, E_Q)：

- V_Q = {v_obj, v_attr1, ..., v_attrK, v_act, v_target, v_state}
- v_obj：目标对象类别节点（如 "woman", "dog", "apple"）
- v_attr_i：属性节点（颜色、材质、大小等）
- v_act：动作节点（如 "watering", "running"）
- v_target：动作/关系目标对象
- v_state：状态节点（如 "half-eaten", "sitting"）

- E_Q = {(v_obj, r_has, v_attr_i), (v_obj, r_doing, v_act), (v_obj, r_spatial, v_target), ...}
- r ∈ R = {has, wearing, doing, behind, on, inside, holding, ...} 预定义关系类型集合

**定义 2 (视觉推理增强开放词汇分割)**：给定图像 I 和查询图 G_Q，目标是预测二元分割掩码 M，使得 M_{ij}=1 当且仅当像素(i,j)对应的对象同时满足 G_Q 中所有约束。

**定义 3 (组合深度层级)**：
- **L1-属性级**：对象类别 + 1~2个属性（如 "red car", "wooden chair"）
- **L2-空间级**：对象 + 空间关系（如 "person behind the table"）
- **L3-动作级**：对象 + 动作/交互（如 "dog running", "man reading book"）
- **L4-嵌套级**：3+个约束组合（如 "woman in red dress watering flowers behind the fence"）

---

## 2. 系统架构设计

### 2.1 整体架构

```
输入图像 I                    输入查询 Q
+----------+          +----------------------+
|  SAM2    |          |  [模块A] LLM查询解析   |
| Hiera-L  |          |  +- LLM -> 结构化JSON  |
| Backbone |          |  +- BEiT-3 文本编码    |
| (冻结)   |          |  +- GNN 图推理         |
+----+-----+          +----------+-----------+
     |                           |
     v                      查询图 G_Q = (V,E)
+------------------+       节点特征 f_v in R^256
| [模块B] 场景图   |       边特征   f_e in R^768
| 增强视觉编码     |              |
|                  |              |
| +- 多尺度FPN      |              |
| +- 场景图生成分支  |              |
| +- HOI特征提取     |              |
| +- 空间位置编码    |              |
+--------+---------+              |
         |                        |
         v                        v
+------------------------------------------+
|         [模块C] 组合式特征匹配             |
|                                          |
| +-----------+ +---------------------+   |
| | BCM 类别匹配| | ATTM 属性匹配        |   |
| | 点积相似度 | | 颜色/材质/尺寸投影   |   |
| +-----------+ +---------------------+   |
| +-----------+ +---------------------+   |
| | RSM 关系匹配| | ACMM 动作匹配        |   |
| | 场景图验证 | | HOI特征交叉注意力    |   |
| +-----------+ +---------------------+   |
|                                          |
| +-----------------------------------+   |
| | CMF 跨模态融合 (3层Cross-Attn)     |   |
| +-----------------------------------+   |
+-------------------+----------------------+
                    |
                    v
+------------------------------------------+
|       [模块D] 迭代式精化解码器             |
|                                          |
| Stage 1: Coarse Localization            |
|   SAM2 prompt_encoder + mask_decoder     |
|   基于类别嵌入粗定位候选区域               |
|                                          |
| Stage 2: Attribute Verification         |
|   逐属性约束过滤候选mask                  |
|   属性匹配分数门控                        |
|                                          |
| Stage 3: Relational Refinement          |
|   空间关系+动作约束精化边界               |
|   mask -> visual prompt 二次推理          |
+-------------------+----------------------+
                    |
                    v
              最终Mask输出
```

### 2.2 关键设计原则

1. **模块化可插拔**：A/B/C/D四个模块松耦合，可独立训练和评估
2. **轻量适配**：继承OpenWorldSAM"冻结SAM2骨干+轻量适配"哲学，目标 ~12M 可训练参数
3. **组合泛化**：通过结构化查询图分解实现训练中未见组合的零样本泛化
4. **继承SOTA代码基础**：
   - OpenWorldSAM -> SAM2 backbone + BEiT-3 + CrossAttention + two_stage_inference
   - SGC-Net -> 层次化描述分组 + HOI token + 分层粒度比较
   - Prompt-DINO -> 早期融合BiAttention + 多尺度Deformable Attention
   - Semantic-SAM -> 多粒度mask预测 + many-to-many匹配
   - OVSAM -> SAM2CLIP知识迁移（视觉<->语义对齐）
   - COSINE -> 多模态提示统一（文本+视觉示例）

---

## 3. 核心模块详解

### 3.1 模块A: 复杂查询解析器（LLM-based Query Parser）

**设计依据**：SGC-Net使用LLM递归分组生成层次化描述，证明LLM生成的描述优于纯类别标签。VR-OV将此思路扩展为全面的结构化查询图解析。

#### 3.1.1 核心流程

**Step 1 - LLM解析（DeepSeek-V3 API, JSON mode）**：
- 输入： "the woman in red dress who is watering flowers"
- Prompt： "Parse this visual query into a structured JSON with: target_object(category), attributes[type,value], action[verb,target], spatial_relation[type,reference], state"
- 输出JSON：
```json
{
  "target_object": {"name": "woman", "category": "person"},
  "attributes": [
    {"type": "clothing", "value": "dress"},
    {"type": "color", "value": "red"}
  ],
  "action": {"verb": "watering", "target": "flowers"},
  "spatial_relation": null,
  "state": null
}
```

**Step 2 - 文本编码（BEiT-3 Text Encoder，冻结，来自OpenWorldSAM）**：
- f_v_obj = BEiT3("woman") -> R^768
- f_v_attr_color = BEiT3("red color") -> R^768
- f_v_act = BEiT3("watering") -> R^768
- f_v_tgt = BEiT3("flowers") -> R^768

**Step 3 - 图推理（2层GAT，4头注意力）**：
- 节点特征经MLP投影：768 -> 256
- 关系类型用可学习嵌入（参考SGC-Net prefix tokens）
- GAT消息传递：h_v^(l+1) = sigma(sum(alpha_uv * W^(l) * h_u^l))
- 输出：上下文增强的节点特征 F_query = {v: h_v^final in R^256}

#### 3.1.2 伪代码实现

```python
class LLMQueryParser(nn.Module):
    def __init__(self):
        super().__init__()
        # BEiT-3文本编码器 (冻结，来自OpenWorldSAM)
        self.text_encoder = BEiT3TextEncoder.from_pretrained("beit3_large")
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # 可学习关系类型嵌入 (~50种)
        self.relation_embed = nn.Embedding(50, 768)

        # 节点特征投影: 768 -> 256
        self.node_proj = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 256)
        )

        # GNN图推理 (2层GAT, 4头)
        self.gnn = QueryGraphGAT(
            in_dim=256, hidden_dim=256, num_layers=2, num_heads=4
        )

    def forward(self, query_text: str):
        # Step 1: LLM解析 (离线，支持缓存)
        parsed = self._llm_parse(query_text)

        # Step 2: 逐节点文本编码
        node_texts = self._extract_node_texts(parsed)
        node_feats = {}
        for node_id, text in node_texts.items():
            tokens = self.text_encoder.tokenize(text)
            feat = self.text_encoder.encode(tokens)  # [768]
            node_feats[node_id] = self.node_proj(feat)  # [256]

        # Step 3: 构建图 + GNN推理
        edge_index, edge_types = self._build_edges(parsed)
        edge_attr = self.relation_embed(edge_types)  # [E, 768]
        enhanced = self.gnn(node_feats, edge_index, edge_attr)

        return {
            "node_features": enhanced,
            "edge_index": edge_index,
            "target_node": "obj"
        }
```

#### 3.1.3 可训练组件明细

| 组件 | 参数量 | 来源参考 |
|------|--------|----------|
| relation_embed | 50x768 = 38.4K | SGC-Net prefix tokens |
| node_proj MLP | ~590K | OpenWorldSAM text_hidden_fcs |
| QueryGraphGAT (2层) | ~2M | 标准GAT |
| BEiT-3 Text Encoder | 冻结 | OpenWorldSAM |
| **模块A总计** | **~2.6M** | |

---

### 3.2 模块B: 场景图增强视觉编码（Scene-Graph Enhanced Visual Encoder）

**设计依据**：
- OpenWorldSAM: SAM2 Hiera-L backbone + BEiT-3多模态提取器架构
- SGC-Net: HOI token + Cross Attention 提取交互特征 + gate_weight门控
- Prompt-DINO: 多尺度Deformable Attention + 早期融合BiAttention

#### 3.2.1 核心流程

SAM2 Hiera-L Backbone (冻结) 输出多尺度特征图：
- F_1: [B, 144, H/8, W/8]
- F_2: [B, 288, H/16, W/16]
- F_3: [B, 576, H/32, W/32]

新增场景图分支（可训练 ~5.8M）：
1. 输入投影: Conv1x1 (C->256) x3层（参考OpenWorldSAM _bb_feat_sizes）
2. Region Proposal Head: Conv3x3->Conv1x1输出[objectness,dx,dy]，Top-K=64
3. Relation Prediction Head: 参考SGC-Net HOIResidualAttentionBlock
   - [HOI] x [PATCH] Cross Attention
   - 预测9种空间关系: {above, below, left, right, inside, behind, in_front_of, on, holding}
4. HOI Feature Extractor: 5个可学习HOI tokens（参考SGC-Net）
5. 场景图特征注入 + 门控融合（可学习gate参考SGC-Net）

#### 3.2.2 伪代码实现

```python
class SceneGraphVisualEncoder(nn.Module):
    def __init__(self, sam2_backbone):
        super().__init__()
        self.backbone = sam2_backbone  # 冻结

        self.input_proj = nn.ModuleList([
            nn.Conv2d(144, 256, 1),
            nn.Conv2d(288, 256, 1),
            nn.Conv2d(576, 256, 1),
        ])

        self.region_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(), nn.Conv2d(128, 3, 1)
        )

        self.hoi_tokens = nn.Parameter(
            torch.randn(5, 256) * (256 ** -0.5)
        )
        self.hoi_cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)

        self.sg_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1)
        )

        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, image):
        with torch.no_grad():
            backbone_out = self.backbone.forward_image(image)
            _, img_emb, _, _ = self.backbone._prepare_backbone_features(backbone_out)

        feat_256 = [proj(f) for proj, f in zip(self.input_proj, img_emb)]
        low_feat = feat_256[-1]

        feat_flat = low_feat.flatten(2).permute(0,2,1)
        hoi_tok = self.hoi_tokens.unsqueeze(0).expand(low_feat.shape[0],-1,-1)
        hoi_feat, _ = self.hoi_cross_attn(hoi_tok, feat_flat, feat_flat)

        enhanced = []
        for feat_i in feat_256:
            hoi_spatial = hoi_feat.mean(1).unsqueeze(-1).unsqueeze(-1)
            hoi_spatial = hoi_spatial.expand(-1,-1,
                feat_i.shape[-2], feat_i.shape[-1])
            cat = torch.cat([feat_i, hoi_spatial], dim=1)
            sg_enh = self.sg_conv(cat)
            g = self.gate.sigmoid()
            enhanced.append(g * feat_i + (1-g) * sg_enh)

        return {
            "enhanced_features": enhanced,
            "vanilla_features": img_emb,
            "hoi_features": hoi_feat,
        }
```

#### 3.2.3 可训练组件明细

| 组件 | 参数量 | 来源参考 |
|------|--------|----------|
| input_proj (3xConv1x1) | ~0.5M | OpenWorldSAM |
| region_head | ~0.3M | 简单卷积头 |
| hoi_tokens | 5x256=1.3K | SGC-Net |
| hoi_cross_attn (MHA) | ~1.6M | SGC-Net |
| sg_conv | ~1.2M | 场景图注入 |
| gate | 1 | SGC-Net |
| **模块B总计** | **~5.8M** | |

---

### 3.3 模块C: 组合式特征匹配（Compositional Feature Matcher）

**设计依据**：
- OpenWorldSAM: CrossAttention Transformer (VLM特征 <-> 视觉特征交互)
- Prompt-DINO: BiAttentionBlock 早期融合实现文本-视觉双向增强
- SGC-Net: 分层粒度比较 (Hierarchical Group Comparison)

#### 3.3.1 四路匹配 + 跨模态融合

架构概览：
1. **BCM (Basic Category Matching)**: f_obj (dot) visual_feat -> M_cat [B,H,W]。使用点积+可学习温度参数
2. **ATTM (Attribute Matching)**: 颜色/材质/尺寸属性感知投影匹配 -> M_attr [B,H,W]
3. **RSM (Relational Spatial Matching)**: 查询图关系 vs 场景图边特征 -> M_rel [B,H,W]
4. **ACMM (Action Matching)**: 动作查询 vs HOI特征交叉注意力 -> M_act [B,H,W]
5. **CMF (Cross-Modal Fusion)**: 3层Cross-Attention融合 -> F_composed [N_nodes, 256]

#### 3.3.2 伪代码实现

```python
class CompositionalFeatureMatcher(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()

        # BCM
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07))

        # ATTM: 属性感知投影
        self.color_proj = nn.Linear(d_model, 64)
        self.material_proj = nn.Linear(d_model, 64)
        self.color_text_proj = nn.Linear(256, 64)
        self.material_text_proj = nn.Linear(256, 64)

        # RSM: 关系匹配MLP
        self.relation_matcher = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # ACMM: 动作匹配
        self.action_matcher = nn.MultiheadAttention(d_model, 8, batch_first=True)

        # CMF: 跨模态融合 (3层Cross-Attention)
        self.cmf = CrossAttentionFusion(
            d_model=d_model, num_heads=8, num_layers=3, dropout=0.1
        )
        self.fusion_proj = nn.Linear(d_model+4, d_model)

    def forward(self, query_features, visual_features, hoi_features):
        B = visual_features[0].shape[0]
        vis_feat = visual_features[-1]  # [B,256,H,W]
        H, W = vis_feat.shape[-2:]

        # 1. BCM: 类别匹配
        obj_feat = F.normalize(query_features["obj"], dim=-1)
        vis_flat = F.normalize(vis_feat.flatten(2), dim=1)
        M_cat = torch.einsum('d,bdh->bh', obj_feat, vis_flat)
        M_cat = M_cat.view(B,H,W) * self.logit_scale.exp()

        # 2. ATTM: 属性匹配
        M_attr = torch.zeros(B,H,W, device=vis_feat.device)
        if "attr_color" in query_features:
            cq = self.color_text_proj(query_features["attr_color"])
            cv = self.color_proj(vis_feat.permute(0,2,3,1))
            M_attr += F.cosine_similarity(cv, cq.unsqueeze(0).unsqueeze(0), dim=-1)
        if "attr_material" in query_features:
            mq = self.material_text_proj(query_features["attr_material"])
            mv = self.material_proj(vis_feat.permute(0,2,3,1))
            M_attr += F.cosine_similarity(mv, mq.unsqueeze(0).unsqueeze(0), dim=-1)

        # 3. RSM: 关系匹配
        M_rel = torch.zeros(B,H,W, device=vis_feat.device)

        # 4. ACMM: 动作匹配
        M_act = torch.zeros(B,H,W, device=vis_feat.device)
        if "action" in query_features:
            act_feat = query_features["action"]
            attn_out, _ = self.action_matcher(
                act_feat.unsqueeze(0).unsqueeze(0).expand(B,-1,-1),
                hoi_features, hoi_features
            )
            M_act = attn_out.mean(-1).unsqueeze(-1).unsqueeze(-1).expand(B,H,W)

        # 5. CMF: 跨模态融合
        match_stack = torch.stack([M_cat,M_attr,M_rel,M_act], dim=-1)
        vis_flat = vis_feat.flatten(2).permute(0,2,1)
        match_flat = match_stack.view(B,-1,4)
        cmf_input = self.fusion_proj(
            torch.cat([vis_flat, match_flat], dim=-1)
        )

        query_stack = torch.stack([
            q for q in query_features.values() if q.dim() <= 1
        ]).unsqueeze(0).expand(B,-1,-1)

        composed = self.cmf(query_stack, cmf_input)
        final = composed.mean(dim=1)

        return {
            "composed_features": final,
            "match_maps": {"cat":M_cat,"attr":M_attr,"rel":M_rel,"act":M_act},
            "per_node_features": composed
        }
```

#### 3.3.3 可训练组件明细

| 组件 | 参数量 | 来源参考 |
|------|--------|----------|
| 属性投影 (4xLinear) | ~66K | 属性感知 |
| relation_matcher (MLP) | ~0.3M | SGC-Net |
| action_matcher (MHA) | ~0.26M | 标准MHA |
| CMF (3层Cross-Attn) | ~3M | OpenWorldSAM |
| fusion_proj | ~67K | 维度对齐 |
| **模块C总计** | **~3.7M** | |

---

### 3.4 模块D: 迭代式精化解码器（Iterative Refinement Decoder）

**设计依据**：
- OpenWorldSAM: two_stage_inference（粗预测->mask作为visual prompt->精化）
- Semantic-SAM: 多粒度输出（不同level对应不同语义层次）
- Prompt-DINO: MaskDINO decoder + contrastive head

#### 3.4.1 三阶段精化流程

```
F_composed [B, 256]
     |
     v
+----------------------------------------------------------+
| Stage 1: Coarse Localization (类别粗定位)                  |
|                                                           |
| F_composed -> 复制N个positional_tokens (N=100)             |
|  -> SAM2 prompt_encoder(text_embeds)                      |
|  -> SAM2 mask_decoder -> M_coarse [B, N, H/4, W/4]       |
|  -> IoU预测 -> Top-K筛选 (K=10~20)                         |
|  -> 候选区域: {R_1, R_2, ..., R_K}                        |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
| Stage 2: Attribute Verification (属性验证)                 |
|                                                           |
| 对每个候选区域 R_k:                                         |
|   score_attr(R_k) = avg(match_maps["attr"] in R_k)       |
|   门控: keep if score_attr > learnable_threshold          |
|  -> 过滤后候选 {R'_1, ..., R'_M}, M <= K                  |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
| Stage 3: Relational Refinement (关系精化)                   |
|                                                           |
| 用过滤后mask作为visual prompt:                              |
|   SAM2 prompt_encoder(masks=filtered_masks)               |
|   SAM2 mask_decoder 二次推理                                |
|                                                           |
| 综合分数:                                                   |
|   score_final = w1*score_mask + w2*score_attr             |
|                + w3*score_rel + w4*score_act              |
|   w = softmax(learnable_score_weights)                    |
|                                                           |
| -> 输出: 最高分mask作为最终预测                               |
+----------------------------------------------------------+
```

#### 3.4.2 伪代码实现

```python
class IterativeRefinementDecoder(nn.Module):
    def __init__(self, sam2_prompt_encoder, sam2_mask_decoder,
                 num_tokens=100, query_dim=256):
        super().__init__()
        self.prompt_encoder = sam2_prompt_encoder
        self.mask_decoder = sam2_mask_decoder
        self.num_tokens = num_tokens

        # 可学习位置token (参考OpenWorldSAM)
        self.positional_tokens = nn.Parameter(
            torch.randn(num_tokens, query_dim)
        )

        # 可学习门控阈值
        self.attr_threshold = nn.Parameter(torch.tensor(0.3))

        # 分数聚合权重
        self.score_weights = nn.Parameter(torch.ones(4) / 4)

    def forward(self, composed_feat, match_maps,
                image_embeddings, high_res_feats,
                hoi_features=None, query_features=None):
        B = composed_feat.shape[0]

        # Stage 1: Coarse Localization
        feat_with_pos = (
            composed_feat.unsqueeze(1).expand(-1, self.num_tokens, -1) +
            self.positional_tokens.unsqueeze(0)
        )
        sparse_emb, dense_emb = self.prompt_encoder(
            points=None, boxes=None, masks=None,
            text_embeds=feat_with_pos
        )
        low_res_masks, iou_pred, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, repeat_image=True,
            high_res_features=high_res_feats
        )

        # Top-K
        iou_scores = iou_pred.squeeze(-1)
        topk = min(20, self.num_tokens)
        _, topk_idx = iou_scores.topk(topk, dim=1)
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, topk)
        masks_topk = low_res_masks[batch_idx, topk_idx]

        # Stage 2: Attribute Verification
        M_attr = match_maps["attr"]
        M_attr_low = F.interpolate(
            M_attr.unsqueeze(1), size=masks_topk.shape[-2:], mode='bilinear'
        ).squeeze(1)
        attr_scores = (masks_topk.sigmoid() * M_attr_low.unsqueeze(1)
                      ).sum(dim=[-2,-1]) / (
                      masks_topk.sigmoid().sum(dim=[-2,-1]) + 1e-8)
        attr_mask = attr_scores > self.attr_threshold

        # Stage 3: Relational Refinement
        final_outputs = []
        for b in range(B):
            keep = attr_mask[b]
            b_masks = masks_topk[b][keep]

            sp2, d2 = self.prompt_encoder(
                points=None, boxes=None, masks=b_masks.unsqueeze(0),
                text_embeds=None
            )
            ref_masks, ref_iou, _, _ = self.mask_decoder(
                image_embeddings=image_embeddings[b:b+1],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp2,
                dense_prompt_embeddings=d2,
                multimask_output=False, repeat_image=True,
                high_res_features=[f[b:b+1] for f in high_res_feats]
            )

            w = self.score_weights.softmax(dim=0)
            combined = w[0]*ref_iou.squeeze(-1) + w[1]*attr_scores[b][keep]
            final_outputs.append(ref_masks[0, combined.argmax()])

        return {
            "pred_masks": torch.stack(final_outputs),
            "coarse_masks": masks_topk
        }
```

#### 3.4.3 可训练组件明细

| 组件 | 参数量 | 来源参考 |
|------|--------|----------|
| positional_tokens | 100x256=25.6K | OpenWorldSAM |
| attr_threshold | 1 | 可学习门控 |
| score_weights | 4 | 分数聚合 |
| **模块D总计** | **~25.6K** | |

---

### 3.5 VR-OV总计可训练参数

| 模块 | 参数量 | 占比 |
|------|--------|------|
| 模块A (查询解析器) | ~2.6M | 21.5% |
| 模块B (场景图编码器) | ~5.8M | 47.9% |
| 模块C (组合特征匹配) | ~3.7M | 30.5% |
| 模块D (精化解码器) | <0.1M | 0.1% |
| **总计** | **~12.1M** | **100%** |

vs OpenWorldSAM的4.5M，在可控范围内增加2.7倍，但实现了从简单类别查询到复杂组合查询的范式升级。

---

## 4. 训练策略

### 4.1 训练数据构建

#### 4.1.1 基础数据源

| 数据集 | 用途 | 规模 |
|--------|------|------|
| COCO Panoptic 2017 | 基础分割能力训练 | 118K图像 |
| RefCOCO/RefCOCO+/RefCOCOg | 指代分割能力训练 | 26K/20K/26K |
| Visual Genome | 场景图标注 (对象+属性+关系) | 108K图像 |
| HICO-DET / SWIG-HOI | HOI检测标注 | ~38K图像 |
| PACO-LVIS / COCO-Attributes | 物体属性标注 | ~40K图像 |

#### 4.1.2 合成复杂查询数据生成 (核心创新)

利用LLM为现有标注自动生成复杂组合查询，按四个层级构建训练数据：

**生成策略 1 - L1属性级（基于COCO + COCO-Attributes + LLM增强）**：
- 输入：{category, attributes, bbox, mask}
- LLM生成多样化自然语言查询包含属性的不同组合
- 示例：coco标注 "red car" -> "the bright red sports car parked on the street"
- 预计生成：~50K queries

**生成策略 2 - L2空间级（基于Visual Genome场景图 + LLM增强）**：
- 输入：场景图三元组 (obj1, relation, obj2)
- LLM生成包含空间关系的自然语言查询
- 示例：(man, standing behind, counter) -> "the man standing behind the wooden counter"
- 预计生成：~40K queries

**生成策略 3 - L3动作级（基于HICO-DET/SWIG-HOI HOI标注 + LLM增强）**：
- 输入：(person, riding, horse)
- LLM生成包含动作描述的自然语言查询
- 示例：-> "the person who is riding the brown horse"
- 预计生成：~30K queries

**生成策略 4 - L4嵌套级（多数据集组合 + LLM生成）**：
- 从同一图像中选取有关系的多个标注对象
- 组合属性+关系+动作生成L4级复杂查询
- 示例："the woman in blue dress who is sitting on the chair and reading a book"
- 预计生成：~20K queries（需额外人工校验）

| 生成层级 | 数据来源 | 预计数量 | 用途 |
|----------|----------|----------|------|
| L1-属性级 | COCO + COCO-Attributes + LLM增强 | ~50K | 属性匹配训练 |
| L2-空间级 | Visual Genome场景图 + LLM增强 | ~40K | 关系匹配训练 |
| L3-动作级 | HICO-DET/SWIG-HOI + LLM增强 | ~30K | 动作匹配训练 |
| L4-嵌套级 | 多数据集组合 + LLM生成 | ~20K | 组合泛化训练 |
| **总计** | | **~140K** | |

### 4.2 损失函数设计

VR-OV采用多任务联合损失：

**L_total = L_mask + L_attr + L_rel + L_act + L_compose**

具体设计：

```python
# 1. Mask损失 (继承OpenWorldSAM)
L_mask = lambda_bce * BCE(mask_pred, mask_gt)
       + lambda_dice * Dice(mask_pred, mask_gt)

# 2. 属性匹配损失 (新增)
L_attr = CrossEntropy(attr_pred, attr_gt)
# attr_pred: 每个区域的属性分类 [K, N_attr_types]
# attr_gt: 从LLM解析中获取的属性标签

# 3. 关系匹配损失 (新增)
L_rel = BCE(rel_pred, rel_gt)
# rel_pred: 场景图中预测的关系 [K, K, N_rels]
# rel_gt: Visual Genome场景图标注

# 4. 动作匹配损失 (新增)
L_act = ContrastiveLoss(hoi_features, action_query)
# 正样本: 匹配的HOI特征-动作描述对
# 负样本: 随机采样不匹配对（同批次内）

# 5. 组合匹配损失 (新增)
L_compose = TripletLoss(
    composed_features, positive_mask, negative_mask
)
# 确保组合特征与正确mask的相似度 > 与错误mask的相似度
```

损失权重配置：
- lambda_bce = 5.0, lambda_dice = 5.0 (来自OpenWorldSAM)
- lambda_attr = 1.0, lambda_rel = 1.0, lambda_act = 0.5, lambda_compose = 2.0

### 4.3 训练流程

**Phase 1: 单模块预训练 (Warm-up)**

```
Phase 1a: 模块B预训练 (2 epochs)
  - 冻结: SAM2 backbone, 模块A, C, D
  - 训练: 场景图生成分支, HOI提取器
  - 数据: Visual Genome场景图标注
  - 损失: L_rel (关系预测)
  - 学习率: lr=1e-4

Phase 1b: 模块C预训练 (2 epochs)
  - 冻结: SAM2 backbone, 模块A, D
  - 微调: 模块B + 模块C
  - 数据: COCO + 属性标注 + L1合成数据
  - 损失: L_attr + L_act
  - 学习率: lr=5e-5

Phase 1c: 模块A集成 (1 epoch)
  - 冻结: SAM2 backbone
  - 微调: 模块A + B + C
  - 数据: 合成复杂查询 (L1+L2+L3混合)
  - 损失: L_attr + L_rel + L_act + L_compose
  - 学习率: lr=5e-5
```

**Phase 2: 端到端联合训练 (Main Training)**

```
完整系统训练 (20 epochs)
  - 冻结: SAM2 Hiera-L backbone, BEiT-3 text encoder
  - 训练: 模块A(除BEiT-3), B, C, D
  - 优化器: AdamW (lr=1e-4, weight_decay=0.05)
  - 学习率调度: CosineAnnealing + 2 epoch warmup
  - Batch size: 8 (单GPU, 参考OpenWorldSAM)
  - 数据混合比例: L1:L2:L3:L4 = 2:2:1:1
```

**Phase 3: 精调 (Fine-tuning)**

```
针对性精调 (5 epochs)
  - 降低学习率: lr=1e-5
  - 增加L4嵌套查询比例: L1:L2:L3:L4 = 1:1:1:2
  - 引入难负样本挖掘 (Hard Negative Mining)
  - 使用梯度累积模拟更大的batch size
```

### 4.4 训练技巧

1. **梯度控制**：模块D使用detach隔离模块C的梯度，防止Stage 2/3反向传播影响特征提取
2. **渐进式难度**：训练从L1逐步过渡到L4，类似课程学习（Curriculum Learning）
3. **查询图Dropout**：训练时随机丢弃查询图部分边/节点(prob=0.2)，增强鲁棒性
4. **多尺度训练**：随机缩放+裁剪，尺寸范围 [384, 1024]（参考OpenWorldSAM）
5. **属性级难负样本挖掘**：同类别但属性不匹配的实例采样为负样本
6. **EMA (Exponential Moving Average)**：维护模型权重EMA版本用于推理

---

## 5. 实验设计

### 5.1 基准测试构建: VR-OV-Bench

需要构建一个新的benchmark来系统评估复杂组合查询的分割能力。

#### 5.1.1 数据组成

| 子集 | 来源 | 样本数 | 层级 | 描述 |
|------|------|--------|------|------|
| VR-OV-Attr | COCO val + 属性标注 | 3000 | L1 | 属性级查询 |
| VR-OV-Spatial | Visual Genome test | 2000 | L2 | 空间关系查询 |
| VR-OV-Action | HICO-DET test + SWIG-HOI | 2000 | L3 | 动作/交互查询 |
| VR-OV-Nested | 多数据集组合+人工标注 | 1000 | L4 | 嵌套组合查询 |
| VR-OV-ZeroShot | 跨数据集采样 | 2000 | L1-L4 | 零样本组合泛化 |
| **总计** | | **~10000** | | |

#### 5.1.2 评估指标

- **mIoU**：标准分割指标（主指标）
- **cIoU (Compositional IoU)**：仅当所有约束满足时才计为正的组合IoU
- **Attr-Acc**：属性匹配准确率（颜色、材质、尺寸等分别报告）
- **Rel-Acc**：关系匹配准确率
- **Comp-Score**：组合分数 = cIoU * min(Attr-Acc, Rel-Acc, Act-Acc)（综合指标）

### 5.2 消融实验方案

#### 5.2.1 模块消融

| 实验ID | 模块A | 模块B | 模块C | 模块D | 描述 |
|--------|-------|-------|-------|-------|------|
| E0 | - | - | - | - | 基线 (OpenWorldSAM原始) |
| E1 | A | - | - | - | +LLM解析器 |
| E2 | - | B | - | - | +场景图分支 |
| E3 | A | B | - | D(Stage1) | 无组合匹配 |
| E4 | A | B | C | D(Stage1) | 仅粗定位无精化 |
| E5 | A | B | C | D(Stage1+2) | +属性验证精化 |
| E6 | A | B | C | D(Full) | **完整VR-OV** |

#### 5.2.2 模块C子组件消融

| 实验ID | BCM | ATTM | RSM | ACMM | CMF | 描述 |
|--------|-----|------|-----|------|-----|------|
| C0 | Y | - | - | - | - | 仅类别匹配 |
| C1 | Y | Y | - | - | - | +属性匹配 |
| C2 | Y | Y | Y | - | - | +关系匹配 |
| C3 | Y | Y | - | Y | - | +动作匹配 |
| C4 | Y | Y | Y | Y | - | 四路匹配无融合 |
| C5 | Y | Y | Y | Y | Y | **完整模块C** |

#### 5.2.3 关键超参数消融

| 实验ID | 参数 | 探究范围 |
|--------|------|----------|
| H1 | LLM模型选择 | DeepSeek-V3 / GPT-4o / Qwen2.5 / 无LLM(规则解析) |
| H2 | 查询图GNN层数 | 1 / 2 / 3 / 4 |
| H3 | N_tokens (模块D) | 50 / 100 / 200 / 300 |
| H4 | CMF层数 | 1 / 2 / 3 / 4 |
| H5 | 训练数据混合比例 | {2:2:1:1, 1:1:1:1, 2:1:1:2, 1:2:2:3} |

### 5.3 对比基线

| 方法 | 类型 | 对比意义 |
|------|------|----------|
| OpenWorldSAM (NeurIPS 2025) | OV统一分割 | 当前SOTA，仅支持简单类别查询 |
| COSINE (ICCV 2025) | 多模态提示统一 | 文本+视觉提示，对比多模态理解 |
| Prompt-DINO (ICCV 2025) | OV检测分割统一 | 早期融合的代表作 |
| SAM2 + CLIP ensemble | 简单组合 | SAM2分割 + CLIP属性分类的简单事后组合 |
| LISA / GroundingDINO | 开放词汇检测 | 检测级定位能力对比 |
| Semantic-SAM | 多粒度分割 | 对比多粒度输出能力 |

### 5.4 预期实验结果（假设性）

| 方法 | L1 (Attr) | L2 (Spatial) | L3 (Action) | L4 (Nested) | Avg cIoU |
|------|-----------|-------------|-------------|-------------|----------|
| OpenWorldSAM | 35.2 | 12.1 | 8.4 | 3.2 | 14.7 |
| SAM2+CLIP ensemble | 42.5 | 18.3 | 11.2 | 5.8 | 19.5 |
| COSINE | 48.1 | 22.6 | 15.8 | 9.1 | 23.9 |
| VR-OV (Ours) | **62.3** | **45.7** | **38.2** | **28.5** | **43.7** |

> 注：以上为基于现有各方法能力的合理预期，实际数值需实验验证。

---

## 6. 预期贡献与创新点

### 创新点 1：首次定义并解决"复杂组合查询的开放词汇统一分割"问题

**创新性**：现有所有OV统一框架（OpenWorldSAM, COSINE, Prompt-DINO等）仅支持简单类别查询。VR-OV首次将开放词汇分割扩展到需要同时理解对象类别、属性、动作、空间关系和状态的复杂组合查询。

**技术突破**：提出结构化查询图作为统一表示形式，将自然语言查询的"复杂度"通过图结构显式编码，使模型可以系统化地分解和处理多维约束。

### 创新点 2：LLM驱动的查询图解析与GNN推理机制

**创新性**：首次将LLM作为"语义编译器"引入开放词汇分割，将自由形式自然语言自动解析为结构化查询图（而非简单的文本嵌入）。创新性地使用GNN进行查询图内部推理，使属性、动作、关系等约束在查询侧就完成语义交互。

**与SGC-Net的区别**：SGC-Net用LLM生成层次化类别描述用于分类，VR-OV用LLM生成包含多维约束的查询图用于分割，范式和粒度完全不同。

### 创新点 3：场景图增强视觉编码的轻量实现

**创新性**：提出在SAM2冻结骨干上增加仅5.8M参数的场景图生成分支（参考SGC-Net HOI token设计），通过门控机制将场景图信息注入视觉特征。这使得视觉编码器天然包含对象间关系信息，显著区别于仅建模独立对象的传统OV分割方法。

**技术优越性**：5.8M参数（仅占SAM2 902M的0.6%）即实现了场景级别的结构化理解。

### 创新点 4：四维组合式特征匹配框架（BCM+ATTM+RSM+ACMM+CMF）

**创新性**：首次提出分维度、分层级的组合特征匹配框架。将类别、属性、关系、动作四个维度的语义约束分别用不同的匹配机制处理，再通过Cross-Attention进行跨模态融合，实现了细粒度的组合语义匹配。

**与Prompt-DINO早期融合的区别**：Prompt-DINO的早期融合是全局的文本-视觉特征交互，VR-OV是分维度的结构化匹配+融合，能更精准地定位每个约束维度对应的视觉证据。

### 创新点 5：三阶段迭代精化解码器

**创新性**：提出"粗定位→属性验证→关系精化"的级联解码结构，每个阶段增加更细粒度的约束条件。创新性地将OpenWorldSAM的two_stage_inference扩展为属性感知的三阶段精化，通过可学习门控阈值实现自适应过滤。

**关键设计**：Stage 2/3使用模块C的匹配分数图作为额外约束信号，实现了"视觉推理"引导的迭代精化。

### 创新点 6：面向组合泛化的合成数据生成策略

**创新性**：提出基于LLM的四级合成数据生成pipeline（L1属性级 + L2空间级 + L3动作级 + L4嵌套级），利用现有标注自动生成~140K复杂组合查询训练样本。这种数据生成策略成本极低，可规模化。

**意义**：使得在无需人工标注复杂组合查询数据的情况下，模型能够学习多维约束的组合推理能力。

### 创新点 7：VR-OV-Bench基准测试

**贡献**：构建首个系统评估复杂组合查询分割能力的benchmark（~10K样本，覆盖4个层级），为后续研究提供标准化的评估平台。

---

## 7. 潜在挑战与应对策略

### 挑战 1：LLM解析的可靠性

**问题**：LLM可能产生不准确或不一致的查询图解析，尤其在歧义查询或边缘案例中。

**应对策略**：
- (a) Prompt Engineering：设计详细的few-shot prompt，提供解析示例
- (b) 解析验证：训练小型验证模型检测明显错误的解析结果
- (c) 容错机制：训练时使用查询图Dropout（随机丢弃部分边），使模型对解析不完美具有鲁棒性
- (d) 对比实验：对比LLM解析 vs 规则解析的效果差异，评估LLM解析的必要性

### 挑战 2：属性匹配的可区分性

**问题**：颜色、材质等细粒度属性的视觉信号可能很弱。例如，"red car"和"dark red car"的视觉差异可能被模型忽略。

**应对策略**：
- (a) 对比学习：使用属性级别的对比损失增强属性特征的判别性
- (b) 数据增强：通过颜色扰动/材质变换等数据增强强调属性差异
- (c) 属性分类预训练：在COCO-Attributes/VG-Attributes上预训练属性分类器
- (d) 多尺度属性匹配：在不同分辨率下分别匹配颜色（全局）、材质（局部）、尺寸（相对）等属性

### 挑战 3：组合泛化的困难

**问题**：训练中未出现的属性-关系-动作组合在测试时可能难以泛化。

**应对策略**：
- (a) 查询图分解训练：训练时随机独立采样各维度的约束，创造更多组合样本
- (b) 因子化建模：将组合匹配分数设计为各维度分数的乘积结构，支持因子化泛化
- (c) Structure-aware数据增强：在语义图中交换属性/关系来生成新的组合
- (d) 元学习训练：设计组合泛化的元学习训练策略

### 挑战 4：计算效率

**问题**：12.1M可训练参数 + LLM API调用 + 三阶段推理可能带来计算开销。

**应对策略**：
- (a) LLM解析缓存：相同的查询文本可缓存解析结果，推理时无需重复调用LLM
- (b) 知识蒸馏：训练后将LLM的解析能力蒸馏到小型Transformer，实现本地化解析
- (c) 推理优化：Stage 1的Top-K筛选后，Stage 2/3只在候选区域进行精化，减少计算量
- (d) 模型剪枝/量化：对训练好的模块进行剪枝和INT8量化
- (e) 与现有框架对比：12.1M vs OpenWorldSAM 4.5M增加了2.7倍参数，但在同等参数量级的OV方法中仍属轻量（Prompt-DINO >100M可训练参数）

### 挑战 5：评估基准的构建

**问题**：VR-OV-Bench需要高质量的复杂组合查询标注，这本身具有挑战性。

**应对策略**：
- (a) 半自动构建：自动化生成 + 人工抽检验证（抽检20%）
- (b) 多轮迭代：第一轮自动生成 -> 人工修正 -> LLM学习修正模式 -> 第二轮自动修正
- (c) 参考RefCOCO：借鉴RefCOCO的标注流程和验证机制
- (d) 分阶段发布：先发布L1-L3子集，L4嵌套级作为挑战集后续发布

---

## 8. 时间线估算

### 总体时间：6个月（1人全职或2人协作）

```
Month 1-2: 基础框架搭建与数据准备
  Week 1-2:
    - 基于OpenWorldSAM代码建立基础训练框架
    - 实现模块D的基础版本 (Stage 1粗定位)
    - 环境搭建、SAM2/BEiT-3权重加载验证

  Week 3-4:
    - 实现模块A (LLM查询解析器 + 文本编码 + GNN)
    - LLM解析pipeline搭建与测试
    - 属性匹配子模块 (ATTM) 实现

  Week 5-6:
    - 实现模块B (场景图分支 + HOI特征提取)
    - 基于Visual Genome的场景图预训练

  Week 7-8:
    - 实现模块C完整版本 (BCM+ATTM+RSM+ACMM+CMF)
    - 合成数据生成pipeline (L1-L4)
    - 数据生成批量运行

Month 3-4: 训练与消融实验
  Week 9-10:
    - 模块B和C的单独预训练
    - 损失函数验证和调优

  Week 11-12:
    - Phase 1完整预训练
    - 初步端到端训练调试验证

  Week 13-14:
    - Phase 2端到端联合训练
    - 消融实验 (E0-E6, C0-C5)

  Week 15-16:
    - Phase 3精调
    - 超参数消融 (H1-H5)
    - 在现有benchmark上评估baseline

Month 5: VR-OV-Bench构建与对比实验
  Week 17-18:
    - VR-OV-Bench构建 (数据收集、LLM生成、人工校验)
    - 在VR-OV-Bench上运行所有baseline

  Week 19-20:
    - 完整对比实验
    - 错误分析、可视化、case study
    - 各层级难度分析

Month 6: 论文撰写与补充实验
  Week 21-22:
    - 论文撰写 (Introduction, Related Work, Method)
    - 补充实验（如有审稿人可能关注的点）

  Week 23-24:
    - 论文撰写 (Experiments, Conclusion)
    - 图表精修、rebuttal准备
    - 代码整理与开源准备
```

### 关键里程碑

| 时间 | 里程碑 | 交付物 |
|------|--------|--------|
| Month 1 结束 | 模块A+D完成 | 可运行的查询解析+粗分割 |
| Month 2 结束 | 全模块完成+数据就绪 | VR-OV完整代码 + ~140K合成数据 |
| Month 3 结束 | 预训练完成 | 消融实验初步结果 |
| Month 4 结束 | 主训练完成 | 主要实验结果表格 |
| Month 5 结束 | Benchmark+对比实验完成 | VR-OV-Bench + 对比结果 |
| Month 6 结束 | 论文完成 | 完整论文草稿 + 代码发布 |

### 资源需求

| 资源 | 规格 | 用途 |
|------|------|------|
| GPU | 1x A100 (80GB) | 训练 (单GPU, batch_size=8) |
| GPU | 1x A6000 (48GB) | 推理评估 + 消融实验并行 |
| LLM API | DeepSeek-V3 | 数据生成 + 查询解析 |
| 存储 | ~500GB | 数据集 + 模型checkpoint |
| 内存 | 128GB | 数据预处理 |

---

## 附录A: 代码实现规划

### A.1 项目结构

```
VR-OV/
├── configs/                    # 配置文件（继承OpenWorldSAM风格）
│   ├── vr_ov_base.yaml
│   └── vr_ov_full.yaml
├── model/
│   ├── __init__.py
│   ├── vr_ov.py               # 主模型 (类似open_world_sam2.py)
│   ├── query_parser.py         # 模块A
│   ├── scene_graph_encoder.py  # 模块B
│   ├── compositional_matcher.py # 模块C
│   ├── refinement_decoder.py   # 模块D
│   ├── cross_attention.py      # CMF (复用OpenWorldSAM)
│   ├── gnn.py                  # GAT图推理
│   └── sam2/                   # SAM2子模块 (来自OpenWorldSAM)
├── data/
│   ├── synthetic_query_gen.py  # 合成数据生成
│   └── vr_ov_dataset.py        # 数据加载
├── train_net.py                # 训练入口
├── eval.py                     # 评估脚本
├── demo.py                     # 演示脚本
└── tools/
    ├── build_vr_ov_bench.py    # Benchmark构建
    └── analyze_errors.py       # 错误分析
```

### A.2 从现有代码的复用关系

| 组件 | 复用来源 | 复用方式 |
|------|----------|----------|
| SAM2 Hiera-L backbone | OpenWorldSAM model/segment_anything_2/ | 直接复用（冻结） |
| BEiT-3 text encoder | OpenWorldSAM model/unilm/beit3/ | 直接复用（冻结） |
| SAM2 prompt_encoder | OpenWorldSAM sam2/modeling/sam/prompt_encoder.py | 直接复用（冻结） |
| SAM2 mask_decoder | OpenWorldSAM sam2/modeling/sam/mask_decoder.py | 直接复用（冻结） |
| CrossAttentionTransformer | OpenWorldSAM model/open_world_sam2.py | 参考实现 |
| two_stage_inference | OpenWorldSAM model/open_world_sam2.py | 参考实现 |
| HOI Residual Attention | SGC-Net models/model.py | 参考实现 |
| gate_weight 门控 | SGC-Net models/model.py | 参考设计 |
| prefix/conjunction tokens | SGC-Net models/model.py | 参考设计 |
| 分层描述生成 | SGC-Net build_tree/ | 参考pipeline |
| 早期融合 BiAttention | Prompt-DINO groundingdino/fuse_modules.py | 参考实现 |
| 多尺度 Deformable Attention | Prompt-DINO pixel_decoder/ | 参考实现 |
| MaskDINO decoder结构 | Prompt-DINO transformer_decoder/ | 参考实现 |
| ContrastiveHead | Prompt-DINO transformer_decoder/maskdino_decoder.py | 参考实现 |
| 多粒度mask | Semantic-SAM architectures/interactive_mask_dino.py | 参考设计 |
| Many-to-many匹配 | Semantic-SAM modules/many2many_matcher.py | 参考设计 |
| SAM2CLIP知识迁移 | OVSAM projects/rwkvsam/models/detectors/ | 参考设计 |

---

## 附录B: 与现有方法的核心区别总结

| 维度 | OpenWorldSAM | COSINE | Prompt-DINO | SGC-Net | **VR-OV (Ours)** |
|------|-------------|--------|-------------|---------|-------------------|
| 查询类型 | 简单类别 | 文本+视觉 | 文本+视觉 | HOI类别 | **复杂组合查询** |
| 属性理解 | 无 | 隐式 | 无 | 部分(类别描述) | **显式属性匹配** |
| 空间关系 | 无 | 无 | 无 | 隐式(检测框) | **显式关系匹配** |
| 动作理解 | 无 | 无 | 无 | 有(HOI) | **显式动作匹配** |
| 多模态提示 | 仅文本 | 文本+视觉 | 文本+视觉 | 仅文本+层次描述 | **文本+查询图** |
| LLM使用 | 无 | 无 | 无 | 层次化描述生成 | **结构化查询图解析** |
| 解码方式 | 单阶段/两阶段 | 单阶段 | 单阶段 | 单阶段 | **三阶段迭代精化** |
| 可训练参数 | 4.5M | ~30M | >100M | ~50M | **~12M** |

---

*文档版本: v1.0 | 日期: 2026-04-27 | 作者: VR-OV Research Team*
