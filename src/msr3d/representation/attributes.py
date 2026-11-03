from typing import Dict, List, Optional, Tuple

import torch

from .types import FeatureSourcePool, GeomGuardRecord, InstanceMetaRecord, PrototypeMetaRecord


def infer_source_type(source_name: str, metadata: Optional[Dict] = None) -> str:
    metadata = metadata or {}
    explicit = str(metadata.get("source_type", "")).strip().lower()
    if explicit in {"mask", "mask_based", "render", "render_based"}:
        return "render" if explicit.startswith("render") else "mask"
    name = str(source_name).lower()
    if "render" in name or "whitebg" in name:
        return "render"
    return "mask"


def _metric_status_high_better(value: Optional[float], safe: float, moderate_low: float) -> str:
    if value is None:
        return "safe"
    if value >= safe:
        return "safe"
    if value >= moderate_low:
        return "moderate"
    return "hard"


def _metric_status_low_better(value: Optional[float], safe_high: float, moderate_high: float) -> str:
    if value is None:
        return "safe"
    if value <= safe_high:
        return "safe"
    if value <= moderate_high:
        return "moderate"
    return "hard"


def classify_geom_guard(obs: Dict, source_type: str, gamma: float = 0.70, eta: float = 0.25) -> GeomGuardRecord:
    visibility = obs.get("visibility_ratio")
    area = obs.get("area_ratio")
    in_frame = obs.get("in_frame_ratio")
    edge = obs.get("edge_contact_ratio", obs.get("edge_truncation"))

    metric_status = {}
    if source_type == "mask":
        metric_status["visibility_ratio"] = _metric_status_high_better(visibility, safe=0.35, moderate_low=0.20)
    metric_status["area_ratio"] = _metric_status_high_better(area, safe=0.0020, moderate_low=0.0005)
    metric_status["in_frame_ratio"] = _metric_status_high_better(in_frame, safe=0.85, moderate_low=0.60)
    metric_status["edge_contact_ratio"] = _metric_status_low_better(edge, safe_high=0.15, moderate_high=0.35)

    hard_bad_flags = {k: (v == "hard") for k, v in metric_status.items()}
    moderate_count = sum(1 for v in metric_status.values() if v == "moderate")
    has_hard = any(hard_bad_flags.values())
    if has_hard or moderate_count >= 2:
        label = "bad"
        multiplier = float(eta)
    elif moderate_count == 1:
        label = "suspicious"
        multiplier = float(gamma)
    else:
        label = "normal"
        multiplier = 1.0
    return GeomGuardRecord(
        frame_name=str(obs.get("frame_name", "")),
        label=label,
        multiplier=float(multiplier),
        moderate_count=int(moderate_count),
        hard_bad_flags=hard_bad_flags,
        metric_status=metric_status,
        source_type=source_type,
    )


def _obs_value_from_tensor(v: Optional[torch.Tensor], idx: int) -> Optional[float]:
    if v is None:
        return None
    if int(v.shape[0]) <= idx:
        return None
    return float(v[idx].item())


def build_geom_guard_for_source(
    source: FeatureSourcePool,
    frame_names: List[str],
    gamma: float = 0.70,
    eta: float = 0.25,
) -> List[GeomGuardRecord]:
    source_type = infer_source_type(source.source_name, source.metadata)
    obs_stats = source.obs_stats or {}
    visibility = obs_stats.get("visibility_ratio")
    area = obs_stats.get("area_ratio")
    in_frame = obs_stats.get("in_frame_ratio")
    edge = obs_stats.get("edge_contact_ratio", obs_stats.get("edge_truncation"))
    out = []
    for i, frame_name in enumerate(frame_names):
        obs = {
            "frame_name": frame_name,
            "visibility_ratio": _obs_value_from_tensor(visibility, i),
            "area_ratio": _obs_value_from_tensor(area, i),
            "in_frame_ratio": _obs_value_from_tensor(in_frame, i),
            "edge_contact_ratio": _obs_value_from_tensor(edge, i),
        }
        item = classify_geom_guard(obs, source_type=source_type, gamma=gamma, eta=eta)
        out.append(item)
    return out


def build_proto_meta(
    source_name: str,
    prototype_stats: List[Dict],
    assign: torch.Tensor,
    geom_guard: List[GeomGuardRecord],
    total_obs_count: int,
    obs_source_names: Optional[List[str]] = None,
    all_source_names: Optional[List[str]] = None,
    total_source_count: int = 1,
) -> Tuple[List[PrototypeMetaRecord], InstanceMetaRecord]:
    geom_bad = torch.tensor([1 if g.label == "bad" else 0 for g in geom_guard], dtype=torch.float32)
    total_obs_count = max(1, int(total_obs_count))
    if all_source_names:
        canonical_source_names = [str(x) for x in all_source_names]
        total_source_count = max(1, int(len(canonical_source_names)))
    else:
        total_source_count = max(1, int(total_source_count))
        canonical_source_names = [str(source_name)]
    if obs_source_names is None:
        obs_source_names = [str(source_name)] * int(assign.shape[0])
    if int(len(obs_source_names)) != int(assign.shape[0]):
        raise ValueError(
            f"obs_source_names length mismatch: {len(obs_source_names)} vs {int(assign.shape[0])}"
        )
    proto_meta: List[PrototypeMetaRecord] = []

    for proto_idx, stat in enumerate(prototype_stats):
        member_idx = torch.nonzero(assign == proto_idx, as_tuple=False).view(-1)
        member_source_names = [str(obs_source_names[int(i)]) for i in member_idx.tolist()]
        if int(member_idx.numel()) > 0:
            geom_bad_fraction = float(geom_bad.index_select(0, member_idx).mean().item())
        else:
            geom_bad_fraction = 0.0
        support_ratio = float(stat.get("assigned_count", 0) / total_obs_count)
        source_counts = {name: 0 for name in canonical_source_names}
        for name in member_source_names:
            source_counts[name] = source_counts.get(name, 0) + 1
        present_count = sum(1 for _name, count in source_counts.items() if count > 0)
        cross_source_support_ratio = float(present_count / total_source_count) if stat.get("assigned_count", 0) > 0 else 0.0
        denom = max(1, int(len(member_source_names)))
        source_comp = [float(source_counts.get(name, 0) / denom) for name in canonical_source_names]
        source_comp_map = {str(name): float(value) for name, value in zip(canonical_source_names, source_comp)}
        proto_meta.append(
            PrototypeMetaRecord(
                proto_index=int(proto_idx),
                support_ratio=support_ratio,
                cross_source_support_ratio=cross_source_support_ratio,
                proto_member_sim_mean=float(stat.get("proto_to_member_sim_mean", 0.0)),
                proto_separation=float(stat.get("proto_separation", 0.0)),
                geom_bad_fraction=geom_bad_fraction,
                source_comp=source_comp,
                source_comp_names=list(canonical_source_names),
                source_comp_map=source_comp_map,
            )
        )

    support_values = [float(x.support_ratio) for x in proto_meta]
    cross_source_values = [float(x.cross_source_support_ratio) for x in proto_meta]
    weighted_cross_source_values = [float(x.support_ratio) * float(x.cross_source_support_ratio) for x in proto_meta]
    bad_geom_proto_ratio = 0.0
    if proto_meta:
        bad_geom_proto_ratio = float(sum(1 for x in proto_meta if float(x.geom_bad_fraction) > 0.5) / len(proto_meta))
    instance_meta = InstanceMetaRecord(
        proto_count=int(len(proto_meta)),
        top1_support_ratio=float(max(support_values) if support_values else 0.0),
        cross_source_agreement_score=float(sum(weighted_cross_source_values)) if weighted_cross_source_values else 0.0,
        bad_geom_proto_ratio=bad_geom_proto_ratio,
        metadata={
            "mean_cross_source_support_ratio": float(sum(cross_source_values) / len(cross_source_values))
            if cross_source_values
            else 0.0,
            "source_names": list(canonical_source_names),
        },
    )
    return proto_meta, instance_meta
