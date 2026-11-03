from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PrototypeConfig:
    normalize_input: bool = True
    kmin: int = 2
    kmax: int = 6
    max_prototypes: Optional[int] = None
    cover_eps: float = 0.003
    cover_theta: float = 0.94
    medoid_weighted: bool = False
    gain_threshold: float = 0.003
    min_gain_to_continue: Optional[float] = None
    redundancy_lambda: float = 0.0
    merge_similarity_threshold: float = 0.985
    prune_min_mass: int = 1
    prune_min_support: Optional[int] = None
    prune_min_ratio: float = 0.0
    geom_guard_gamma: float = 0.70
    geom_guard_eta: float = 0.25

@dataclass
class ReliabilityConfig:
    enabled: bool = False
    score_mode: str = "mean_similarity"
    keep_ratio: float = 0.7
    min_keep: int = 3
    min_score: Optional[float] = None

@dataclass
class FusionConfig:
    normalize_input: bool = True
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)

@dataclass
class Stage3Config:
    stage3_impl: str = "rowwise_agg"
    row_scorer_type: str = "linear"
    row_hidden_dim: int = 32
    row_dropout: float = 0.0
    aggregator_topk: int = 2
    instance_refiner_type: str = "linear"
    instance_refiner_hidden_dim: int = 16
    instance_feature_version: str = "v1"
    use_instance_interactions: bool = True
    instance_refine_train_mode: str = "all"
    max_proto_count: int = 6
    source_names: List[str] = field(default_factory=list)
    use_mode_head: bool = False
    mode_head_type: str = "linear"
    mode_head_hidden_dim: int = 8
    mode_head_score_temperature: float = 1.0
