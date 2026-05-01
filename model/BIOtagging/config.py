from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # Data
    train_queries_path: Path | None = None
    val_queries_path: Path | None = None
    synthetic_queries_count: int = 5000
    val_split_ratio: float = 0.1

    # Model
    hidden_dim: int = 768
    num_tags: int = 14
    num_decoder_layers: int = 2
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 30
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "outputs"
    )
    model_save_name: str = "parser_head_best.pt"

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
