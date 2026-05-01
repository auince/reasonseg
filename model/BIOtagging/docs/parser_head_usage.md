# Parser Head 使用权衡文档

## 1. 是什么

`QueryParserHead` 是一个轻量级 Transformer Encoder，对 BEiT3 的文本隐状态逐 token 做 14 类 BIO 标签预测。

**输入**：BEiT3 编码后的文本隐状态 `[1, seq_len, 1024]`
**输出**：每个 token 的 14 类标签 logits `[1, seq_len, 14]`

## 2. 当前最优 checkpoint

| checkpoint | 路径 |
|---|---|
| **当前最优** | `model/BIOtagging/outputs/stage1_fast_train50k_plus_mask20k_20260429_170500/parser_head_best.pt` |
| 备选（flash+pro 试验） | `model/BIOtagging/outputs/stage1_fast_train50k_plus_mask20k_plus_flashpro2k_20260430_031500/parser_head_best.pt` |

## 3. 独立 Python 加载 + 推理

```python
import torch
import sys
from pathlib import Path

ROOT = Path("/home/lch/Project/ReasonSeg")
sys.path.insert(0, str(ROOT))

from model.BIOtagging.query_parser_head import QueryParserHead

# ── 构建模型（需与训练配置一致） ──
model = QueryParserHead(
    hidden_dim=1024,       # 与 BEiT3 输出维一致
    num_tags=14,
    num_layers=1,          # 当前训练的配置
    nhead=8,
    dim_feedforward=1024,
    dropout=0.1,
)

# ── 加载权重 ──
ckpt_path = ROOT / "model/BIOtagging/outputs/stage1_fast_train50k_plus_mask20k_20260429_170500/parser_head_best.pt"
model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
model.eval()

# ── 推理 ──
# beit3_hidden: BEiT3 编码器的输出，shape [batch, seq_len, 1024]
# attention_mask: [batch, seq_len]，True=有效 token
with torch.no_grad():
    logits = model(beit3_hidden, attention_mask=attention_mask)
    tag_ids = model.predict_tags(beit3_hidden, attention_mask=attention_mask)
    structure = model.decode_structure(
        tokens=["the", "red", "cup"],
        beit3_hidden=beit3_hidden,
        attention_mask=attention_mask,
    )
```

## 4. 完整 pipeline 集成（config 方式）

### 4.1 修改 YAML config

在 `configs/refcoco/refcoco_base.yaml` 中修改：

```yaml
MODEL:
  OpenWorldSAM2:
    LEARNED_PARSER:
      ENABLED: true
      CHECKPOINT: "model/BIOtagging/outputs/stage1_fast_train50k_plus_mask20k_20260429_170500/parser_head_best.pt"
      HIDDEN_DIM: 1024   # 必须与 checkpoint 训练时一致
      NUM_LAYERS: 1      # 必须与 checkpoint 训练时一致
      DROPOUT: 0.1
```

### 4.2 运行时加载流程

`open_world_sam2.py:258-281` 的自动加载逻辑：

```
LEARNED_PARSER.ENABLED == true
  → 构建 QueryParserHead(hidden_dim, num_layers, nhead, ...)
  → torch.load(CHECKPOINT, weights_only=True)
  → parser_head.load_state_dict(state)
```

### 4.3 推理时的调用点

`open_world_sam2.py:426-429`：

```python
if self.learned_parser_enabled and self.parser_head is not None:
    encoder_out = output["encoder_out"]          # [B, seq_len, 1024]
    tag_logits = self.parser_head(encoder_out)    # [B, seq_len, 14]
```

## 5. API 参考

### 5.1 QueryParserHead

```python
class QueryParserHead(nn.Module):
    def __init__(
        self, hidden_dim=1024, num_tags=14, num_layers=1,
        nhead=8, dim_feedforward=1024, dropout=0.1,
    ) -> None: ...

    def forward(
        self, beit3_hidden: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """返回 logits，shape [batch, seq_len, 14]"""

    def predict_tags(
        self, beit3_hidden: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """返回标签 id，shape [batch, seq_len]"""

    def decode_structure(
        self, tokens: list[str], beit3_hidden: Tensor,
        attention_mask: Tensor | None = None,
    ) -> NormalizedQuery:
        """返回结构化查询（target, attributes, relations, actions, ...）"""
```

### 5.2 NormalizedQuery 结构

```python
{
    "target": "cup" | None,              # 主指代对象
    "attributes": ["red", "small"],      # 属性
    "relations": [{"type": "on", "target": "table"}],  # 空间关系
    "actions": [{"verb": "holding", "target": "phone"}],  # 动作
    "negatives": [],                     # 否定标识
    "exists": True,                      # 对象是否存在
}
```

### 5.3 BIO 标签全集（14 类）

| idx | tag | 含义 |
|-----|-----|------|
| 0 | O | 非结构化 token |
| 1 | B-TGT | 目标起始 |
| 2 | I-TGT | 目标继续 |
| 3 | B-ATTR | 属性起始 |
| 4 | I-ATTR | 属性继续 |
| 5 | B-REL | 关系词起始 |
| 6 | I-REL | 关系词继续 |
| 7 | B-REL-TGT | 关系目标起始 |
| 8 | I-REL-TGT | 关系目标继续 |
| 9 | B-ACT | 动作词起始 |
| 10 | I-ACT | 动作词继续 |
| 11 | B-ACT-TGT | 动作目标起始 |
| 12 | I-ACT-TGT | 动作目标继续 |
| 13 | B-NEG | 否定词 |

## 6. 训练/评估命令参考

### 训练（使用已生成的 silver 数据）

```bash
conda run -n reasonseg-py311 python scripts/train_parser_head_stage1_fast.py \
    --silver-path model/BIOtagging/data/expanded_silver_train50k_plus_mask20k.json \
    --output-dir model/BIOtagging/outputs/stage1_fast_my_run \
    --device 0 --max-epochs 30 --batch-size 256 --lr 1e-3
```

### 评估（在原始 3k 验证切片上）

```bash
conda run -n reasonseg-py311 python -c "
import random, torch
from pathlib import Path
from scripts.train_parser_head_stage1_fast import (
    load_silver_data, load_beit3_and_tokenizer, precompute_hidden_states,
    build_labels, build_length_bucketed_loader, QueryParserHead,
    BEIT3_HIDDEN_DIM, N_TAGS, DEFAULT_LABELS, DEFAULT_QUERIES, ID_TO_BIO_TAG,
)
ROOT = Path('.')
device = torch.device('cuda:0')
all_silvers, all_queries = load_silver_data(DEFAULT_LABELS, DEFAULT_QUERIES)
pairs = list(zip(all_queries, all_silvers))
random.Random(42).shuffle(pairs)
all_queries = [q for q,_ in pairs]
all_silvers = [s for _,s in pairs]
val_size = min(300, max(1, len(all_queries) // 10))
val_queries = all_queries[-val_size:]
val_silvers = all_silvers[-val_size:]

beit3, tokenizer = load_beit3_and_tokenizer()
val_hidden = precompute_hidden_states(beit3, tokenizer, val_queries, device, 24)
val_labels = build_labels(val_queries, val_silvers, tokenizer, 24)

model = QueryParserHead(
    hidden_dim=BEIT3_HIDDEN_DIM, num_tags=N_TAGS,
    num_layers=1, nhead=8, dim_feedforward=1024, dropout=0.1,
).to(device)
model.load_state_dict(torch.load(
    'model/BIOtagging/outputs/stage1_fast_train50k_plus_mask20k_20260429_170500/parser_head_best.pt',
    map_location=device, weights_only=True,
))
model.eval()

cm = torch.zeros(N_TAGS, N_TAGS, dtype=torch.long)
correct = total = 0
loader = build_length_bucketed_loader(val_hidden, val_labels, 256, shuffle=False)
with torch.no_grad():
    for h, l in loader:
        h, l = h.to(device), l.to(device)
        preds = model(h).argmax(dim=-1)
        mask = l >= 0
        correct += (preds[mask] == l[mask]).sum().item()
        total += mask.sum().item()
        preds = preds.cpu(); l = l.cpu()
        for p, gt in zip(preds.view(-1), l.view(-1)):
            if gt >= 0:
                cm[gt, p] += 1

print(f'val_acc={correct / max(total, 1):.4f}')
for i in range(N_TAGS):
    tp = cm[i,i].item(); fp = cm[:,i].sum().item() - tp; fn = cm[i,:].sum().item() - tp
    support = int(cm[i,:].sum())
    if support > 0:
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        print(f'{ID_TO_BIO_TAG[i]}\tP={p:.4f}\tR={r:.4f}\tF1={f:.4f}\tSup={support}')
"
```

### 完整 pipeline

```bash
# eval（带 learned parser）
conda run -n reasonseg-py311 python scripts/eval.py \
    --config configs/refcoco/refcoco_reasonseg.yaml \
    --data-root /home/lch/Project/ReasonSeg/datasets \
    --checkpoint /path/to/model_final.pth \
    --split refcoco_val_unc \
    --output-dir /tmp/reasonseg-eval
```

## 7. 模型架构与尺寸

- 1 层 TransformerEncoder
- 8 个注意力头
- FFN 维度：1024
- 参数量：约 6.3M
- checkpoint 大小：约 25 MB（FP32）
- 输入配置（当前最优 checkpoint）：
  - `hidden_dim=1024`
  - `num_layers=1`
  - `nhead=8`
  - `dim_feedforward=1024`
  - `dropout=0.1`

## 8. 重要约束

1. **`hidden_dim` 必须 = 1024**。这是 BEiT3 large 的输出维度。如果 num_layers 或 dropout 与 checkpoint 不一致，`load_state_dict` 会报 key mismatch 错误。
2. **`num_tags` 必须 = 14**。是 `bio_schema.py:BIO_TAGS` 的固定长度。不要随意修改。
3. **batch_first=True**。TransformerEncoderLayer 默认 batch_first=True，forward 输入为 `[B, S, D]`。
4. **attention_mask** 是可选的。传入 `attention_mask` 时，会用 `~mask` 作为 `src_key_padding_mask` 传给 TransformerEncoder。如果不传，所有 token 都参与自注意力。
5. **`weights_only=True`**。`torch.load` 建议总是加这个参数，避免 pickle 安全风险。

## 9. 数据生成流程

生成新训练数据的 scripts 参考：

| script | 用途 |
|---|---|
| `scripts/generate_expanded_silver.py` | 从 RefCOCO pickle + rule parser 生成 `[[query, structure], ...]` |
| `model/BIOtagging/seg_mask_generator.py` | 从 `seg_mask_per_instance.json` bbox/类别合成复杂查询 |
| `scripts/synthesize_reviewed_mask_silver.py` | DeepSeek-V4-Flash 生成 + V4-Pro 审核 |
| `scripts/merge_reviewed_silver.py` | 合并覆盖 silver pool |
