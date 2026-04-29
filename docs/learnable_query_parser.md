# Learnable Query Parser Design

## Motivation

The current `query.py` uses hardcoded word lists for parsing referring expressions:

```python
_RELATION_WORDS = {"behind", "on", "with"}       # 3 words
_ATTRIBUTE_WORDS = {"red", "wooden", "small", "blue"}  # 4 words
_ACTION_TARGETS = {"watering"}                   # 1 word
```

Complex queries like "the second person from the left holding a phone" lose all spatial and action information at parse time, producing a degraded prompt "person" that the model cannot ground.

## Goal

Replace rule-based parsing with a lightweight learned parser that:
1. Preserves the structured `NormalizedQuery` output format
2. Leverages BEiT3 pre-trained knowledge for vocabulary coverage
3. Adds negligible compute (~2M params, ~0.1ms per query)
4. Can be bootstrapped from existing rule outputs as silver labels

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Learnable Query Parser              │
│                                                      │
│  Raw Query: "red cup behind person holding phone"    │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────┐                                 │
│  │ BEiT3 Tokenizer │ → [red, cup, behind, ...]      │
│  └────────┬────────┘                                 │
│           │                                          │
│           ▼                                          │
│  ┌─────────────────┐                                 │
│  │ BEiT3 Encoder   │ → hidden [7 × 768]             │
│  │   (Frozen)       │   (reuses existing backbone)   │
│  └────────┬────────┘                                 │
│           │                                          │
│           ▼                                          │
│  ┌─────────────────┐                                 │
│  │  Parser Head    │ 2-layer Transformer Decoder     │
│  │  (~2M params)   │ + Linear Classification Head    │
│  └────────┬────────┘                                 │
│           │                                          │
│           ▼                                          │
│  BIO Tag Sequence:                                   │
│    red     → B-ATTR                                  │
│    cup     → B-TGT                                   │
│    behind  → B-REL                                   │
│    person  → B-REL-TGT                               │
│    holding → B-ACT                                   │
│    phone   → B-ACT-TGT                               │
│           │                                          │
│           ▼                                          │
│  ┌─────────────────┐                                 │
│  │ Tag → Structure │ deterministic conversion        │
│  │   Converter      │                                 │
│  └────────┬────────┘                                 │
│           │                                          │
│           ▼                                          │
│  NormalizedQuery {                                   │
│    target: "cup",                                    │
│    attributes: ["red"],                              │
│    relations: [{"type": "behind",                    │
│                  "target": "person"}],               │
│    actions: [{"verb": "holding",                     │
│               "target": "phone"}],                   │
│    exists: true                                      │
│  }                                                   │
└──────────────────────────────────────────────────────┘
```

## BIO Tag Schema

| Tag | Meaning | Example |
|-----|---------|---------|
| `O` | Not part of any entity | articles, prepositions |
| `B-TGT` | Beginning of target noun phrase | `cup`, `person` |
| `I-TGT` | Continuation of target | `hot` `dog` |
| `B-ATTR` | Beginning of attribute | `red`, `wooden` |
| `I-ATTR` | Continuation of attribute | `dark` `blue` |
| `B-REL` | Relation word | `behind`, `on`, `left` `of` |
| `I-REL` | Continuation of relation | `of` in `left of` |
| `B-REL-TGT` | Relation target noun phrase | `person` after `behind` |
| `I-REL-TGT` | Continuation of relation target | `hot` `dog` |
| `B-ACT` | Action verb | `holding`, `eating` |
| `I-ACT` | Continuation of action | `looking` `at` |
| `B-ACT-TGT` | Action target noun | `phone` after `holding` |
| `I-ACT-TGT` | Continuation of action target | `red` `ball` |
| `B-NEG` | Negation prefix | `no`, `without` |

## Training Data Generation

Since RefCOCO annotations only provide raw sentences, we generate silver labels:

```python
def generate_training_data(queries: list[str]) -> list[dict]:
    """
    Step 1: Parse with current rule-based parser
    Step 2: Convert structured output to BIO tags
    Step 3: Return (tokens, tags) pairs
    """
    for query in queries:
        parsed = QueryParser().parse(query)  # current rules
        tokens = query.lower().split()
        tags = _structure_to_bio(tokens, parsed)
        yield {"tokens": tokens, "tags": tags}
```

This bootstraps the parser from existing rules, then the model learns to generalize.

## Parser Head Implementation

```python
class QueryParserHead(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_tags: int = 13):
        super().__init__()
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.classifier = nn.Linear(hidden_dim, num_tags)
        self.num_tags = num_tags

    def forward(
        self, beit3_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.transformer(beit3_hidden, beit3_hidden)
        return self.classifier(x)  # [batch, seq_len, num_tags]
```

## Integration Points

### 1. Forward pass hook (inside `_select_prompts`)

```python
# open_world_sam2.py, OpenWorldSAM2 class
def _select_prompts(self, batch_input):
    if self.learned_parser_enabled:
        # Get BEiT3 hidden states (already computed)
        # Run parser head
        # Return composed prompts from learned parse
        ...
```

### 2. Configuration

```yaml
# configs/refcoco/refcoco_base.yaml
MODEL:
  OpenWorldSAM2:
    LEARNED_PARSER:
      ENABLED: true
      NUM_LAYERS: 2
      DROPOUT: 0.1
```

### 3. Training

Option A: Joint training with main model (parser head trained by downstream grounding loss)

Option B: Two-stage training:
1. Train parser head on silver BIO labels
2. Fine-tune end-to-end with grounding task

## Comparison

| | Rule-Based (Current) | Learnable Parser (Proposed) |
|---|---|---|
| Vocabulary | 8 hardcoded words | Full BEiT3 vocabulary (~250K) |
| "left of" | ❌ | ✅ |
| "striped shirt" | ❌ | ✅ |
| "second from right" | ❌ | ✅ (context learning) |
| New spatial terms | Edit source code | Add training examples |
| Parameters | 0 | ~2M |
| Inference latency | ~0 | <0.1ms per query |
| Training required | No | Yes (silver label bootstrapping) |
| Output format | `NormalizedQuery` | `NormalizedQuery` (same) |

## Implementation Plan

### Phase 1: Parser Head Module
- [ ] Create `reasonseg/query_parser_head.py`
- [ ] Implement `QueryParserHead` class
- [ ] Implement BIO tag schema and `_structure_to_bio` converter
- [ ] Implement `_bio_to_structure` reverse converter

### Phase 2: Training Data
- [ ] Script to generate silver BIO labels from current rules
- [ ] Train/val split on RefCOCO queries
- [ ] Data loader for parser training

### Phase 3: Integration
- [ ] Add `learned_parser_enabled` flag to model config
- [ ] Hook parser head into `_select_prompts`
- [ ] Train end-to-end with grounding loss

### Phase 4: Evaluation
- [ ] Compare parser accuracy vs rules on held-out queries
- [ ] Benchmark grounding metrics (cIoU, mIoU, prec@0.5)
- [ ] Ablation: parser-only vs end-to-end

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Silver labels propagate rule errors | Human review + iterative refinement |
| Parser head overfits to rule outputs | Add dropout + small training set |
| BEiT3 hidden states not optimal for parsing | Allow BEiT3 to be trainable if needed |
| Inference overhead | <0.1ms per query on GPU, negligible |

## References

- Current rule-based parser: `reasonseg/query.py`
- Current prompt composition: `reasonseg/modeling/prompting.py`
- BEiT3 encoder: `reasonseg/modeling/evf_sam2.py` (mm_extractor)
- Model forward: `reasonseg/modeling/open_world_sam2.py` (_select_prompts, forward)
