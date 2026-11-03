from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

@dataclass
class FeatureSourcePool:
    source_name: str
    frame_features: Dict[str, torch.Tensor] = field(default_factory=dict)
    features: Optional[torch.Tensor] = None
    frame_ids: Optional[torch.Tensor] = None
    frame_names: List[str] = field(default_factory=list)
    obs_stats: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

@dataclass
class InstanceFeaturePool:
    instance_id: str
    sources: Dict[str, FeatureSourcePool]
    metadata: Dict = field(default_factory=dict)

@dataclass
class SharedCandidateRecord:
    global_index: int
    source_name: str
    source_local_index: int
    frame_name: str
    feature: torch.Tensor
    metadata: Dict = field(default_factory=dict)

@dataclass
class MeanRepresentation:
    source_name: str
    mean_feature: torch.Tensor
    frame_names: List[str]
    feature_dim: int
    metadata: Dict = field(default_factory=dict)

@dataclass
class PrototypeRepresentation:
    source_name: str
    prototype_features: torch.Tensor
    prototype_indices: List[int]
    prototype_frame_names: List[str]
    proto_mass: List[float]
    frame_names: List[str]
    feature_dim: int
    prototype_stats: List[Dict] = field(default_factory=list)
    prototype_meta: List["PrototypeMetaRecord"] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ReliabilityFilterResult:
    source_name: str
    frame_names: List[str]
    reliability_scores: List[float]
    keep_mask: List[bool]
    kept_frame_names: List[str]
    dropped_frame_names: List[str]
    metadata: Dict = field(default_factory=dict)

@dataclass
class SourceRepresentationBundle:
    source_name: str
    mean: Optional[MeanRepresentation] = None
    prototype_set: Optional[PrototypeRepresentation] = None
    reliability_filter: Optional[ReliabilityFilterResult] = None
    geom_guard: Optional[List["GeomGuardRecord"]] = None
    instance_meta: Optional["InstanceMetaRecord"] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class GeomGuardRecord:
    frame_name: str
    label: str
    multiplier: float
    moderate_count: int
    hard_bad_flags: Dict[str, bool]
    metric_status: Dict[str, str]
    source_type: str

@dataclass
class PrototypeMetaRecord:
    proto_index: int
    support_ratio: float
    cross_source_support_ratio: float
    proto_member_sim_mean: float
    proto_separation: float
    geom_bad_fraction: float
    source_comp: List[float]
    source_comp_names: List[str]
    source_comp_map: Dict[str, float]
    metadata: Dict = field(default_factory=dict)

@dataclass
class InstanceMetaRecord:
    proto_count: int
    top1_support_ratio: float
    cross_source_agreement_score: float
    bad_geom_proto_ratio: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class QueryInput:
    query_id: str
    text_feature: torch.Tensor
    metadata: Dict = field(default_factory=dict)
