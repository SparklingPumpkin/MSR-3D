from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from .settings import Stage3Config
from .types import InstanceMetaRecord, PrototypeMetaRecord, QueryInput, SourceRepresentationBundle

def _normalize_vector(feat: torch.Tensor) -> torch.Tensor:
    feat = feat.detach().cpu().float().view(-1)
    return F.normalize(feat.view(1, -1), dim=1).squeeze(0)

def _prototype_sims(bundle: SourceRepresentationBundle, text_feat: torch.Tensor) -> torch.Tensor:
    if bundle.prototype_set is None or bundle.prototype_set.prototype_features is None:
        raise ValueError(f"source {bundle.source_name} missing prototype_set representation")
    protos = bundle.prototype_set.prototype_features.detach().cpu().float()
    if protos.ndim != 2 or int(protos.shape[0]) == 0:
        raise ValueError(f"source {bundle.source_name} has empty prototype_set")
    return torch.matmul(protos, text_feat.view(-1, 1)).squeeze(1)

def _proto_meta_vector(
    record: PrototypeMetaRecord,
    source_names: Optional[List[str]] = None,
) -> List[float]:
    if source_names:
        source_comp = [float(record.source_comp_map.get(name, 0.0)) for name in source_names]
    else:
        source_comp = [float(x) for x in record.source_comp]
    return [
        float(record.support_ratio),
        float(record.cross_source_support_ratio),
        float(record.proto_member_sim_mean),
        float(record.proto_separation),
        float(record.geom_bad_fraction),
        *source_comp,
    ]

def _instance_meta_vector(record: InstanceMetaRecord) -> List[float]:
    return [
        float(record.proto_count),
        float(record.top1_support_ratio),
        float(record.cross_source_agreement_score),
        float(record.bad_geom_proto_ratio),
    ]

def build_stage3_inputs(
    bundle: SourceRepresentationBundle,
    query: QueryInput,
    config: Stage3Config,
    normalize_text: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    if bundle.prototype_set is None:
        raise ValueError(f"source {bundle.source_name} missing prototype_set")
    if bundle.instance_meta is None:
        raise ValueError(f"source {bundle.source_name} missing instance_meta")

    text_feat = query.text_feature
    text_feat = _normalize_vector(text_feat) if normalize_text else text_feat.detach().cpu().float().view(-1)
    sims = _prototype_sims(bundle, text_feat)
    proto_meta = list(bundle.prototype_set.prototype_meta)
    if len(proto_meta) != int(sims.numel()):
        raise ValueError(
            f"prototype_meta length mismatch for {bundle.source_name}: "
            f"{len(proto_meta)} vs {int(sims.numel())}"
        )

    max_proto_count = int(config.max_proto_count)
    valid_proto_count = int(sims.numel())
    if valid_proto_count > max_proto_count:
        raise ValueError(
            f"valid proto count exceeds Stage3Config.max_proto_count for {bundle.source_name}: "
            f"{valid_proto_count} > {max_proto_count}"
        )

    canonical_source_names = [str(name) for name in config.source_names if str(name)] if config.source_names else []
    inst_vec = _instance_meta_vector(bundle.instance_meta)
    row_dim = (
        1 + len(_proto_meta_vector(proto_meta[0], source_names=canonical_source_names)) + len(inst_vec) + 1
        if proto_meta
        else 1 + len(inst_vec) + 1
    )
    matrix = torch.zeros((max_proto_count, row_dim), dtype=torch.float32)
    valid_mask = torch.zeros((max_proto_count,), dtype=torch.float32)

    for row_idx, (sim, meta) in enumerate(zip(sims.tolist(), proto_meta)):
        row = [
            float(sim),
            *_proto_meta_vector(meta, source_names=canonical_source_names),
            *inst_vec,
            1.0,
        ]
        matrix[row_idx] = torch.tensor(row, dtype=torch.float32)
        valid_mask[row_idx] = 1.0

    return matrix, valid_mask, {
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "flat_dim": int(matrix.numel()),
        "valid_proto_count": valid_proto_count,
        "padding_proto_count": int(max_proto_count - valid_proto_count),
        "query_id": query.query_id,
        "source_name": bundle.source_name,
        "canonical_source_names": canonical_source_names,
        "prototype_frame_names": list(bundle.prototype_set.prototype_frame_names),
    }
