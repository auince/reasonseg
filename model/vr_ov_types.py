# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class QueryGraph:
    """Graph representation of a parsed referring expression.

    Each node corresponds to a semantic primitive (category, attribute,
    relation, action).  Edges encode syntactic or semantic relations
    between those primitives.
    """

    nodes: List[torch.Tensor]
    edges: torch.Tensor
    node_types: List[str]


@dataclass
class CompositionScores:
    """Composition-matching scores for a single region proposal.

    Each field is a score map ``[B, 1, H, W]`` (or ``None`` if the
    modality is absent from the query).  Attribute scores are further
    split into colour / material / size sub-maps for fine-grained
    supervision.
    """

    cat_feat: Optional[torch.Tensor] = None
    attr_feat: Optional[torch.Tensor] = None
    attr_color: Optional[torch.Tensor] = None
    attr_material: Optional[torch.Tensor] = None
    attr_size: Optional[torch.Tensor] = None
    rel_feat: Optional[torch.Tensor] = None
    act_feat: Optional[torch.Tensor] = None


@dataclass
class RefineState:
    """Snapshot captured at a single iteration of the refine decoder."""

    mask: torch.Tensor
    iou: float
    stage: int
    converged: bool = False


@dataclass
class VR_OV_Output:
    """Complete output of the VR-OV model for one image."""

    pred_masks: torch.Tensor
    comp_scores: CompositionScores
    query_graph: QueryGraph
    refine_history: List[RefineState]
    loss_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
