import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .attributes import build_geom_guard_for_source, build_proto_meta
from .settings import FusionConfig, PrototypeConfig, ReliabilityConfig
from .types import (
    FeatureSourcePool,
    GeomGuardRecord,
    InstanceMetaRecord,
    InstanceFeaturePool,
    MeanRepresentation,
    PrototypeRepresentation,
    ReliabilityFilterResult,
    SharedCandidateRecord,
    SourceRepresentationBundle,
)


def _parse_frame_index(frame_name: str) -> Optional[int]:
    matches = re.findall(r"(\d+)", str(frame_name))
    if not matches:
        return None
    return int(matches[-1])


def _source_count(source: FeatureSourcePool) -> int:
    if source.features is not None:
        return int(source.features.shape[0])
    return int(len(source.frame_features))


def _default_frame_names(n: int, frame_ids: Optional[torch.Tensor]) -> List[str]:
    if frame_ids is not None:
        ids = frame_ids.detach().cpu().view(-1).tolist()
        return [f"frame_{int(x):05d}" for x in ids]
    return [f"frame_{i:05d}" for i in range(n)]


def _effective_kmax(cfg: PrototypeConfig) -> int:
    return int(cfg.max_prototypes) if cfg.max_prototypes is not None else int(cfg.kmax)


def _effective_gain_threshold(cfg: PrototypeConfig) -> float:
    return float(cfg.min_gain_to_continue) if cfg.min_gain_to_continue is not None else float(cfg.gain_threshold)


def _effective_prune_min_support(cfg: PrototypeConfig) -> int:
    return int(cfg.prune_min_support) if cfg.prune_min_support is not None else int(cfg.prune_min_mass)


def _instance_meta_from_repr(proto_repr: PrototypeRepresentation) -> InstanceMetaRecord:
    item = dict(proto_repr.metadata.get("instance_meta", {}))
    return InstanceMetaRecord(
        proto_count=int(item.get("proto_count", 0)),
        top1_support_ratio=float(item.get("top1_support_ratio", 0.0)),
        cross_source_agreement_score=float(item.get("cross_source_agreement_score", 0.0)),
        bad_geom_proto_ratio=float(item.get("bad_geom_proto_ratio", 0.0)),
        metadata=dict(item.get("metadata", {})),
    )


def _stack_source(source: FeatureSourcePool, normalize_input: bool) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    if source.features is not None:
        feats = source.features
        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"source {source.source_name} features is not a Tensor")
        feats = feats.detach().cpu().float()
        if feats.ndim == 1:
            feats = feats.view(1, -1)
        if feats.ndim != 2 or int(feats.shape[0]) == 0:
            raise ValueError(f"source {source.source_name} has invalid aggregated feature shape: {tuple(feats.shape)}")

        n = int(feats.shape[0])
        frame_ids = source.frame_ids
        if frame_ids is not None:
            if not isinstance(frame_ids, torch.Tensor):
                raise TypeError(f"source {source.source_name} frame_ids is not a Tensor")
            frame_ids = frame_ids.detach().cpu().view(-1).long()
            if int(frame_ids.numel()) != n:
                raise ValueError(f"source {source.source_name} frame_ids length mismatch: {int(frame_ids.numel())} vs {n}")
        else:
            parsed = []
            for name in source.frame_names:
                idx = _parse_frame_index(name)
                parsed.append(idx if idx is not None else len(parsed))
            frame_ids = torch.tensor(parsed if parsed else list(range(n)), dtype=torch.long)

        frame_names = list(source.frame_names) if source.frame_names else _default_frame_names(n, frame_ids)
        if len(frame_names) != n:
            raise ValueError(f"source {source.source_name} frame_names length mismatch: {len(frame_names)} vs {n}")
        if normalize_input:
            feats = F.normalize(feats, dim=1)
        return frame_names, feats, frame_ids

    if not source.frame_features:
        raise ValueError(f"source {source.source_name} has no frame features")

    frame_names = sorted(source.frame_features.keys())
    feats = []
    dim = None
    frame_ids_list = []
    for i, frame_name in enumerate(frame_names):
        feat = source.frame_features[frame_name]
        if not isinstance(feat, torch.Tensor):
            raise TypeError(f"frame {frame_name} is not a Tensor")
        feat = feat.detach().cpu().float().view(-1)
        if dim is None:
            dim = int(feat.numel())
        elif int(feat.numel()) != dim:
            raise ValueError(f"inconsistent feature dim in source {source.source_name}: {frame_name}")
        feats.append(feat)
        frame_idx = _parse_frame_index(frame_name)
        frame_ids_list.append(frame_idx if frame_idx is not None else i)
    feats = torch.stack(feats, dim=0)
    if normalize_input:
        feats = F.normalize(feats, dim=1)
    frame_ids = torch.tensor(frame_ids_list, dtype=torch.long)
    return frame_names, feats, frame_ids


def _flatten_instance_sources(
    instance_pool: InstanceFeaturePool,
    normalize_input: bool,
    geom_guard_gamma: float,
    geom_guard_eta: float,
) -> Dict[str, object]:
    if not instance_pool.sources:
        raise ValueError(f"instance {instance_pool.instance_id} has no sources")

    source_names = sorted(instance_pool.sources.keys())
    all_feats: List[torch.Tensor] = []
    all_frame_names: List[str] = []
    all_frame_names_qualified: List[str] = []
    all_frame_ids: List[torch.Tensor] = []
    all_source_ids: List[int] = []
    all_source_names: List[str] = []
    all_source_local_indices: List[int] = []
    all_geom_guard: List[GeomGuardRecord] = []
    shared_candidates: List[SharedCandidateRecord] = []
    source_row_indices: Dict[str, List[int]] = {name: [] for name in source_names}

    global_index = 0
    for source_id, source_name in enumerate(source_names):
        source = instance_pool.sources[source_name]
        frame_names, feats, frame_ids = _stack_source(source, normalize_input=normalize_input)
        geom_guard = build_geom_guard_for_source(
            source,
            frame_names=frame_names,
            gamma=float(geom_guard_gamma),
            eta=float(geom_guard_eta),
        )
        all_feats.append(feats)
        all_frame_names.extend(frame_names)
        all_frame_names_qualified.extend([f"{source_name}::{frame_name}" for frame_name in frame_names])
        all_frame_ids.append(frame_ids)
        all_source_ids.extend([source_id] * int(feats.shape[0]))
        all_source_names.extend([source_name] * int(feats.shape[0]))
        all_source_local_indices.extend(list(range(int(feats.shape[0]))))
        all_geom_guard.extend(geom_guard)

        for local_idx, frame_name in enumerate(frame_names):
            shared_candidates.append(
                SharedCandidateRecord(
                    global_index=global_index,
                    source_name=source_name,
                    source_local_index=int(local_idx),
                    frame_name=str(frame_name),
                    feature=feats[local_idx].detach().cpu(),
                    metadata={"frame_id": int(frame_ids[local_idx].item())},
                )
            )
            source_row_indices[source_name].append(global_index)
            global_index += 1

    flat_feats = torch.cat(all_feats, dim=0)
    flat_frame_ids = torch.cat(all_frame_ids, dim=0)
    flat_source_ids = torch.tensor(all_source_ids, dtype=torch.long)
    guard_multipliers = torch.tensor([float(item.multiplier) for item in all_geom_guard], dtype=flat_feats.dtype)

    return {
        "source_names": source_names,
        "features": flat_feats,
        "frame_names": all_frame_names,
        "qualified_frame_names": all_frame_names_qualified,
        "frame_ids": flat_frame_ids,
        "source_ids": flat_source_ids,
        "source_name_per_obs": all_source_names,
        "source_local_indices": all_source_local_indices,
        "geom_guard": all_geom_guard,
        "guard_multipliers": guard_multipliers,
        "shared_candidates": shared_candidates,
        "source_row_indices": source_row_indices,
    }


def _subset_source(
    source: FeatureSourcePool,
    kept_frame_names: Optional[List[str]] = None,
    keep_mask: Optional[List[bool]] = None,
) -> FeatureSourcePool:
    frame_names, feats, frame_ids = _stack_source(source, normalize_input=False)
    if keep_mask is None:
        if kept_frame_names is None:
            raise ValueError("either kept_frame_names or keep_mask must be provided")
        keep_set = set(kept_frame_names)
        keep_mask = [name in keep_set for name in frame_names]
    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    if not keep_indices:
        keep_indices = [0]

    idx = torch.tensor(keep_indices, dtype=torch.long)
    metadata = dict(source.metadata)
    metadata["filtered_from_frame_count"] = int(len(frame_names))
    metadata["kept_frame_count"] = int(len(keep_indices))

    if source.features is not None:
        obs_stats = {}
        for k, v in source.obs_stats.items():
            if isinstance(v, torch.Tensor) and int(v.shape[0]) == int(len(frame_names)):
                obs_stats[k] = v.detach().cpu().index_select(0, idx)
            else:
                obs_stats[k] = v
        return FeatureSourcePool(
            source_name=source.source_name,
            features=feats.index_select(0, idx),
            frame_ids=frame_ids.index_select(0, idx),
            frame_names=[frame_names[i] for i in keep_indices],
            obs_stats=obs_stats,
            metadata=metadata,
        )

    frame_features = {frame_names[i]: feats[i] for i in keep_indices}
    return FeatureSourcePool(
        source_name=source.source_name,
        frame_features=frame_features,
        frame_names=[frame_names[i] for i in keep_indices],
        frame_ids=frame_ids.index_select(0, idx),
        metadata=metadata,
    )


def _continuity_stats(frame_names: List[str]) -> Dict[str, Optional[float]]:
    frame_indices = []
    for name in sorted(set(frame_names)):
        idx = _parse_frame_index(name)
        if idx is not None:
            frame_indices.append(idx)
    frame_indices = sorted(frame_indices)
    if not frame_indices:
        return {
            "frame_index_min": None,
            "frame_index_max": None,
            "frame_index_span": None,
            "max_consecutive_run": 0,
            "continuity_ratio": 0.0,
            "mean_frame_gap": None,
        }
    if len(frame_indices) == 1:
        return {
            "frame_index_min": int(frame_indices[0]),
            "frame_index_max": int(frame_indices[0]),
            "frame_index_span": 0,
            "max_consecutive_run": 1,
            "continuity_ratio": 1.0,
            "mean_frame_gap": 0.0,
        }
    gaps = [b - a for a, b in zip(frame_indices[:-1], frame_indices[1:])]
    max_run = 1
    cur_run = 1
    for g in gaps:
        if g == 1:
            cur_run += 1
        else:
            max_run = max(max_run, cur_run)
            cur_run = 1
    max_run = max(max_run, cur_run)
    return {
        "frame_index_min": int(frame_indices[0]),
        "frame_index_max": int(frame_indices[-1]),
        "frame_index_span": int(frame_indices[-1] - frame_indices[0]),
        "max_consecutive_run": int(max_run),
        "continuity_ratio": float(max_run / max(1, len(frame_indices))),
        "mean_frame_gap": float(sum(gaps) / max(1, len(gaps))),
    }


def _greedy_cover_gain_selection(
    feats: torch.Tensor,
    cfg: PrototypeConfig,
    guard_multipliers: Optional[torch.Tensor] = None,
    geom_guard: Optional[List[Dict]] = None,
) -> Tuple[List[int], List[Dict], torch.Tensor]:
    n = int(feats.shape[0])
    sim = torch.matmul(feats, feats.t())
    cover = torch.zeros((n,), dtype=feats.dtype)
    selected: List[int] = []
    trace: List[Dict] = []
    if guard_multipliers is None:
        guard_multipliers = torch.ones((n,), dtype=feats.dtype)
    kmax = _effective_kmax(cfg)
    gain_threshold = _effective_gain_threshold(cfg)

    kmin = min(max(1, int(cfg.kmin)), n)
    max_select = min(max(1, kmax), n)
    while len(selected) < max_select:
        gains = torch.clamp(sim - cover.view(-1, 1), min=0.0).mean(dim=0)
        gains = gains * guard_multipliers
        redundancy = torch.zeros_like(gains)
        if selected:
            redundancy = sim[selected, :].max(dim=0).values
        objective = gains - float(cfg.redundancy_lambda) * redundancy
        for idx in selected:
            objective[idx] = -1e9
            gains[idx] = -1e9

        best_idx = int(torch.argmax(objective).item())
        best_gain = float(gains[best_idx].item())
        best_redundancy = float(redundancy[best_idx].item()) if selected else 0.0
        best_objective = float(objective[best_idx].item())

        if selected and len(selected) >= kmin and best_gain < gain_threshold:
            break

        selected.append(best_idx)
        geom_info = geom_guard[best_idx] if geom_guard is not None and 0 <= best_idx < len(geom_guard) else None
        trace.append(
            {
                "selection_rank": int(len(selected)),
                "prototype_source_index": int(best_idx),
                "cover_gain": best_gain,
                "redundancy": best_redundancy,
                "objective_score": best_objective,
                "geom_guard_label": str(getattr(geom_info, "label", "normal")),
                "geom_guard_multiplier": float(getattr(geom_info, "multiplier", 1.0)),
            }
        )
        cover = torch.maximum(cover, sim[:, best_idx])

        if len(selected) >= kmin and float(torch.min(cover).item()) >= float(cfg.cover_theta):
            break

    if not selected:
        selected = [0]
        trace = [
            {
                "selection_rank": 1,
                "prototype_source_index": 0,
                "cover_gain": 0.0,
                "redundancy": 0.0,
                "objective_score": 0.0,
            }
        ]

    return selected, trace, sim


def _greedy_shared_cover_gain_selection(
    feats: torch.Tensor,
    source_ids: torch.Tensor,
    cfg: PrototypeConfig,
    guard_multipliers: Optional[torch.Tensor] = None,
    geom_guard: Optional[List[GeomGuardRecord]] = None,
    source_names: Optional[List[str]] = None,
) -> Tuple[List[int], List[Dict], torch.Tensor]:
    n = int(feats.shape[0])
    sim = torch.matmul(feats, feats.t())
    cover = torch.zeros((n,), dtype=feats.dtype)
    selected: List[int] = []
    trace: List[Dict] = []
    if guard_multipliers is None:
        guard_multipliers = torch.ones((n,), dtype=feats.dtype)
    unique_source_ids = sorted(set(int(x) for x in source_ids.detach().cpu().tolist()))
    source_masks = {sid: (source_ids == sid) for sid in unique_source_ids}
    kmax = _effective_kmax(cfg)
    gain_threshold = _effective_gain_threshold(cfg)
    kmin = min(max(1, int(cfg.kmin)), n)
    max_select = min(max(1, kmax), n)

    while len(selected) < max_select:
        per_source_gains: List[torch.Tensor] = []
        for sid in unique_source_ids:
            rows = source_masks[sid]
            if int(rows.sum().item()) == 0:
                continue
            gains_sid = torch.clamp(sim[rows, :] - cover[rows].view(-1, 1), min=0.0).mean(dim=0)
            per_source_gains.append(gains_sid)
        if not per_source_gains:
            break
        gains = torch.stack(per_source_gains, dim=0).mean(dim=0)
        gains = gains * guard_multipliers
        redundancy = torch.zeros_like(gains)
        if selected:
            redundancy = sim[selected, :].max(dim=0).values
        objective = gains - float(cfg.redundancy_lambda) * redundancy
        for idx in selected:
            objective[idx] = -1e9
            gains[idx] = -1e9

        best_idx = int(torch.argmax(objective).item())
        best_gain = float(gains[best_idx].item())
        best_redundancy = float(redundancy[best_idx].item()) if selected else 0.0
        best_objective = float(objective[best_idx].item())

        if selected and len(selected) >= kmin and best_gain < gain_threshold:
            break

        selected.append(best_idx)
        geom_info = geom_guard[best_idx] if geom_guard is not None and 0 <= best_idx < len(geom_guard) else None
        source_id = int(source_ids[best_idx].item())
        source_name = (
            str(source_names[source_id])
            if source_names is not None and 0 <= source_id < len(source_names)
            else str(source_id)
        )
        trace.append(
            {
                "selection_rank": int(len(selected)),
                "prototype_source_index": int(best_idx),
                "candidate_source_id": int(source_id),
                "candidate_source_name": source_name,
                "cover_gain": best_gain,
                "redundancy": best_redundancy,
                "objective_score": best_objective,
                "geom_guard_label": str(getattr(geom_info, "label", "normal")),
                "geom_guard_multiplier": float(getattr(geom_info, "multiplier", 1.0)),
            }
        )
        cover = torch.maximum(cover, sim[:, best_idx])

        if len(selected) >= kmin and float(torch.min(cover).item()) >= float(cfg.cover_theta):
            break

    if not selected:
        selected = [0]
        source_id = int(source_ids[0].item())
        source_name = (
            str(source_names[source_id])
            if source_names is not None and 0 <= source_id < len(source_names)
            else str(source_id)
        )
        trace = [
            {
                "selection_rank": 1,
                "prototype_source_index": 0,
                "candidate_source_id": int(source_id),
                "candidate_source_name": source_name,
                "cover_gain": 0.0,
                "redundancy": 0.0,
                "objective_score": 0.0,
            }
        ]
    return selected, trace, sim


def _postprocess_prototypes(
    feats: torch.Tensor,
    sim: torch.Tensor,
    proto_indices: List[int],
    selection_trace: List[Dict],
    cfg: PrototypeConfig,
) -> Tuple[List[int], List[Dict]]:
    if len(proto_indices) <= 1:
        return proto_indices, selection_trace

    protos = feats[proto_indices]
    assign = torch.argmax(torch.matmul(feats, protos.t()), dim=1)
    masses = torch.tensor([(assign == k).sum().item() for k in range(len(proto_indices))], dtype=torch.float32)
    ratios = masses / max(1, int(feats.shape[0]))

    order = sorted(range(len(proto_indices)), key=lambda i: (-float(masses[i].item()), i))
    kept_local: List[int] = []
    for local_idx in order:
        if int(masses[local_idx].item()) < _effective_prune_min_support(cfg):
            continue
        if float(ratios[local_idx].item()) < float(cfg.prune_min_ratio):
            continue
        src_idx = proto_indices[local_idx]
        is_dup = False
        for kept_idx in kept_local:
            kept_src_idx = proto_indices[kept_idx]
            if float(sim[src_idx, kept_src_idx].item()) >= float(cfg.merge_similarity_threshold):
                is_dup = True
                break
        if not is_dup:
            kept_local.append(local_idx)

    kmin = max(1, int(cfg.kmin))
    if not kept_local:
        kept_local = [int(torch.argmax(masses).item())]
    if len(kept_local) < kmin and len(order) >= kmin:
        seen = set(kept_local)
        for local_idx in order:
            if local_idx in seen:
                continue
            kept_local.append(local_idx)
            seen.add(local_idx)
            if len(kept_local) >= kmin:
                break

    kept_source_indices = {proto_indices[i] for i in kept_local}
    final_proto_indices = [idx for idx in proto_indices if idx in kept_source_indices]
    trace_map = {int(item["prototype_source_index"]): item for item in selection_trace}
    final_trace = [trace_map[idx] for idx in final_proto_indices]
    return final_proto_indices, final_trace


def _prototype_stats(
    feats: torch.Tensor,
    protos: torch.Tensor,
    assign: torch.Tensor,
    sim: torch.Tensor,
    frame_names: List[str],
    proto_indices: List[int],
    selection_trace: Optional[List[Dict]] = None,
) -> List[Dict]:
    n = int(feats.shape[0])
    k = int(protos.shape[0])
    proto_pairwise = torch.matmul(protos, protos.t()) if k > 0 else None
    trace_map = {
        int(item["prototype_source_index"]): item for item in (selection_trace or [])
    }
    stats: List[Dict] = []
    for proto_idx in range(k):
        member_idx = torch.nonzero(assign == proto_idx, as_tuple=False).view(-1)
        assigned_count = int(member_idx.numel())
        assigned_ratio = float(assigned_count / max(1, n))
        member_frame_names = [frame_names[int(i)] for i in member_idx.tolist()]
        unique_frame_names = sorted(set(member_frame_names))
        assigned_frame_count = int(len(unique_frame_names))
        assigned_frame_ratio = float(assigned_frame_count / max(1, len(set(frame_names))))

        if assigned_count > 0:
            member_sims = sim[member_idx, proto_idx].detach().cpu().float()
            mean_assigned_similarity = float(member_sims.mean().item())
            min_assigned_similarity = float(member_sims.min().item())
            max_assigned_similarity = float(member_sims.max().item())
            proto_to_member_sim_mean = float(member_sims.mean().item())
            proto_to_member_sim_std = float(torch.std(member_sims, unbiased=False).item())
            proto_to_member_sim_min = float(member_sims.min().item())
            proto_to_member_sim_max = float(member_sims.max().item())
            prototype_to_cluster_mean_sim = float(member_sims.mean().item())
        else:
            mean_assigned_similarity = 0.0
            min_assigned_similarity = 0.0
            max_assigned_similarity = 0.0
            proto_to_member_sim_mean = 0.0
            proto_to_member_sim_std = 0.0
            proto_to_member_sim_min = 0.0
            proto_to_member_sim_max = 0.0
            prototype_to_cluster_mean_sim = 0.0

        if k <= 1:
            nearest_proto_sim = 0.0
            proto_separation = 1.0
        else:
            other = torch.cat([proto_pairwise[proto_idx, :proto_idx], proto_pairwise[proto_idx, proto_idx + 1 :]], dim=0)
            nearest_proto_sim = float(torch.max(other).item()) if int(other.numel()) > 0 else 0.0
            proto_separation = float(1.0 - nearest_proto_sim)

        continuity = _continuity_stats(unique_frame_names)
        source_idx = int(proto_indices[proto_idx])
        stats.append(
            {
                "proto_index": int(proto_idx),
                "prototype_source_index": source_idx,
                "prototype_frame_name": str(frame_names[source_idx]),
                "assigned_count": assigned_count,
                "assigned_ratio": assigned_ratio,
                "assigned_frame_count": assigned_frame_count,
                "assigned_frame_ratio": assigned_frame_ratio,
                "mean_assigned_similarity": mean_assigned_similarity,
                "min_assigned_similarity": min_assigned_similarity,
                "max_assigned_similarity": max_assigned_similarity,
                "proto_to_member_sim_mean": proto_to_member_sim_mean,
                "proto_to_member_sim_std": proto_to_member_sim_std,
                "proto_to_member_sim_min": proto_to_member_sim_min,
                "proto_to_member_sim_max": proto_to_member_sim_max,
                "prototype_to_cluster_mean_sim": prototype_to_cluster_mean_sim,
                "nearest_proto_sim": nearest_proto_sim,
                "proto_separation": proto_separation,
                **continuity,
                **trace_map.get(source_idx, {}),
            }
        )
    return stats


def build_mean(source: FeatureSourcePool, config: FusionConfig) -> MeanRepresentation:
    frame_names, feats, _frame_ids = _stack_source(source, normalize_input=bool(config.normalize_input))
    mean_feat = feats.mean(dim=0, keepdim=True)
    mean_feat = F.normalize(mean_feat, dim=1).squeeze(0)
    return MeanRepresentation(
        source_name=source.source_name,
        mean_feature=mean_feat,
        frame_names=frame_names,
        feature_dim=int(mean_feat.numel()),
        metadata={"source_metadata": source.metadata},
    )


def compute_reliability_filter(source: FeatureSourcePool, config: FusionConfig) -> ReliabilityFilterResult:
    cfg: ReliabilityConfig = config.reliability
    frame_names, feats, _frame_ids = _stack_source(source, normalize_input=bool(config.normalize_input))
    n = int(feats.shape[0])
    if cfg.score_mode != "mean_similarity":
        raise ValueError(f"unsupported reliability score_mode: {cfg.score_mode}")

    if n == 1:
        reliability_scores = [1.0]
        keep_mask = [True]
    else:
        sim = torch.matmul(feats, feats.t())
        scores = (sim.sum(dim=1) - 1.0) / max(1, n - 1)
        scores = scores.detach().cpu().float()
        keep_count = min(n, max(int(cfg.min_keep), int(math.ceil(n * float(cfg.keep_ratio)))))
        keep_indices = set(int(i) for i in torch.argsort(scores, descending=True)[:keep_count].tolist())
        if cfg.min_score is not None:
            for i, score in enumerate(scores.tolist()):
                if float(score) >= float(cfg.min_score):
                    keep_indices.add(i)
        keep_mask = [i in keep_indices for i in range(n)]
        reliability_scores = [float(x) for x in scores.tolist()]

    kept_frame_names = [name for name, keep in zip(frame_names, keep_mask) if keep]
    dropped_frame_names = [name for name, keep in zip(frame_names, keep_mask) if not keep]
    if not kept_frame_names:
        kept_frame_names = [frame_names[0]]
        dropped_frame_names = frame_names[1:]
        keep_mask = [i == 0 for i in range(len(frame_names))]

    return ReliabilityFilterResult(
        source_name=source.source_name,
        frame_names=frame_names,
        reliability_scores=reliability_scores,
        keep_mask=keep_mask,
        kept_frame_names=kept_frame_names,
        dropped_frame_names=dropped_frame_names,
        metadata={
            "score_mode": cfg.score_mode,
            "keep_ratio": float(cfg.keep_ratio),
            "min_keep": int(cfg.min_keep),
            "min_score": cfg.min_score,
            "input_frame_count": int(len(frame_names)),
            "kept_frame_count": int(len(kept_frame_names)),
            "dropped_frame_count": int(len(dropped_frame_names)),
        },
    )


def build_prototypes(source: FeatureSourcePool, config: FusionConfig) -> PrototypeRepresentation:
    frame_names, feats, _frame_ids = _stack_source(source, normalize_input=bool(config.normalize_input))
    geom_guard = build_geom_guard_for_source(
        source,
        frame_names=frame_names,
        gamma=float(config.prototype.geom_guard_gamma),
        eta=float(config.prototype.geom_guard_eta),
    )
    guard_multipliers = torch.tensor([float(x.multiplier) for x in geom_guard], dtype=feats.dtype)
    proto_indices, selection_trace, sim_obs = _greedy_cover_gain_selection(
        feats,
        config.prototype,
        guard_multipliers=guard_multipliers,
        geom_guard=geom_guard,
    )
    proto_indices, selection_trace = _postprocess_prototypes(feats, sim_obs, proto_indices, selection_trace, config.prototype)

    protos = F.normalize(feats[proto_indices], dim=1)
    sim = torch.matmul(feats, protos.t())
    assign = torch.argmax(sim, dim=1)
    proto_mass = [float((assign == k).sum().item()) for k in range(int(protos.shape[0]))]
    prototype_frame_names = [frame_names[i] for i in proto_indices]
    prototype_stats = _prototype_stats(
        feats=feats,
        protos=protos,
        assign=assign,
        sim=sim,
        frame_names=frame_names,
        proto_indices=proto_indices,
        selection_trace=selection_trace,
    )
    prototype_meta, instance_meta = build_proto_meta(
        source_name=source.source_name,
        prototype_stats=prototype_stats,
        assign=assign,
        geom_guard=geom_guard,
        total_obs_count=int(len(frame_names)),
        obs_source_names=[source.source_name] * int(len(frame_names)),
        all_source_names=[source.source_name],
        total_source_count=1,
    )
    return PrototypeRepresentation(
        source_name=source.source_name,
        prototype_features=protos,
        prototype_indices=proto_indices,
        prototype_frame_names=prototype_frame_names,
        proto_mass=proto_mass,
        frame_names=frame_names,
        feature_dim=int(protos.shape[1]),
        prototype_stats=prototype_stats,
        prototype_meta=prototype_meta,
        metadata={
            "source_metadata": source.metadata,
            "prototype_stats_version": 2,
            "prototype_meta_version": 1,
            "builder_impl": "greedy_cover_gain_real_observation_only",
            "input_frame_count": int(len(frame_names)),
            "selected_candidate_count": int(len(selection_trace)),
            "final_proto_count": int(len(proto_indices)),
            "prototype_config": {
                "kmin": int(config.prototype.kmin),
                "kmax": _effective_kmax(config.prototype),
                "gain_threshold": _effective_gain_threshold(config.prototype),
                "redundancy_lambda": float(config.prototype.redundancy_lambda),
                "cover_theta": float(config.prototype.cover_theta),
                "merge_similarity_threshold": float(config.prototype.merge_similarity_threshold),
                "prune_min_support": _effective_prune_min_support(config.prototype),
                "prune_min_ratio": float(config.prototype.prune_min_ratio),
                "geom_guard_gamma": float(config.prototype.geom_guard_gamma),
                "geom_guard_eta": float(config.prototype.geom_guard_eta),
            },
            "geom_guard_counts": {
                "normal": int(sum(1 for x in geom_guard if x.label == "normal")),
                "suspicious": int(sum(1 for x in geom_guard if x.label == "suspicious")),
                "bad": int(sum(1 for x in geom_guard if x.label == "bad")),
            },
            "instance_meta": {
                "proto_count": int(instance_meta.proto_count),
                "top1_support_ratio": float(instance_meta.top1_support_ratio),
                "cross_source_agreement_score": float(instance_meta.cross_source_agreement_score),
                "bad_geom_proto_ratio": float(instance_meta.bad_geom_proto_ratio),
                "metadata": dict(instance_meta.metadata),
            },
        },
    )


def build_plain_source_bundle(source: FeatureSourcePool, config: FusionConfig) -> SourceRepresentationBundle:
    prototype_set = build_prototypes(source, config)
    return SourceRepresentationBundle(
        source_name=source.source_name,
        mean=build_mean(source, config),
        prototype_set=prototype_set,
        geom_guard=build_geom_guard_for_source(
            source,
            frame_names=prototype_set.frame_names,
            gamma=float(config.prototype.geom_guard_gamma),
            eta=float(config.prototype.geom_guard_eta),
        ),
        instance_meta=_instance_meta_from_repr(prototype_set),
        metadata={
            "builder_mode": "plain",
            "input_frame_count": _source_count(source),
        },
    )


def build_reliability_guided_source_bundle(source: FeatureSourcePool, config: FusionConfig) -> SourceRepresentationBundle:
    filt = compute_reliability_filter(source, config)
    filtered_source = _subset_source(source, keep_mask=filt.keep_mask)
    prototype_set = build_prototypes(filtered_source, config)
    return SourceRepresentationBundle(
        source_name=source.source_name,
        mean=build_mean(filtered_source, config),
        prototype_set=prototype_set,
        reliability_filter=filt,
        geom_guard=build_geom_guard_for_source(
            filtered_source,
            frame_names=prototype_set.frame_names,
            gamma=float(config.prototype.geom_guard_gamma),
            eta=float(config.prototype.geom_guard_eta),
        ),
        instance_meta=_instance_meta_from_repr(prototype_set),
        metadata={
            "builder_mode": "reliability_guided",
            "input_frame_count": _source_count(source),
            "kept_frame_count": _source_count(filtered_source),
        },
    )


def build_shared_mean(instance_pool: InstanceFeaturePool, config: FusionConfig) -> MeanRepresentation:
    flat = _flatten_instance_sources(
        instance_pool,
        normalize_input=bool(config.normalize_input),
        geom_guard_gamma=float(config.prototype.geom_guard_gamma),
        geom_guard_eta=float(config.prototype.geom_guard_eta),
    )
    feats = flat["features"]
    mean_feat = feats.mean(dim=0, keepdim=True)
    mean_feat = F.normalize(mean_feat, dim=1).squeeze(0)
    return MeanRepresentation(
        source_name="shared",
        mean_feature=mean_feat,
        frame_names=list(flat["qualified_frame_names"]),
        feature_dim=int(mean_feat.numel()),
        metadata={
            "source_names": list(flat["source_names"]),
            "input_frame_count": int(feats.shape[0]),
            "builder_impl": "shared_multisource_mean",
        },
    )


def build_shared_prototypes(instance_pool: InstanceFeaturePool, config: FusionConfig) -> PrototypeRepresentation:
    flat = _flatten_instance_sources(
        instance_pool,
        normalize_input=bool(config.normalize_input),
        geom_guard_gamma=float(config.prototype.geom_guard_gamma),
        geom_guard_eta=float(config.prototype.geom_guard_eta),
    )
    feats = flat["features"]
    source_ids = flat["source_ids"]
    geom_guard = flat["geom_guard"]
    guard_multipliers = flat["guard_multipliers"]
    qualified_frame_names = flat["qualified_frame_names"]
    source_name_per_obs = flat["source_name_per_obs"]
    shared_candidates = flat["shared_candidates"]

    proto_indices, selection_trace, sim_obs = _greedy_shared_cover_gain_selection(
        feats=feats,
        source_ids=source_ids,
        cfg=config.prototype,
        guard_multipliers=guard_multipliers,
        geom_guard=geom_guard,
        source_names=list(flat["source_names"]),
    )
    proto_indices, selection_trace = _postprocess_prototypes(
        feats,
        sim_obs,
        proto_indices,
        selection_trace,
        config.prototype,
    )

    protos = F.normalize(feats[proto_indices], dim=1)
    sim = torch.matmul(feats, protos.t())
    assign = torch.argmax(sim, dim=1)
    proto_mass = [float((assign == k).sum().item()) for k in range(int(protos.shape[0]))]
    prototype_frame_names = [qualified_frame_names[i] for i in proto_indices]
    prototype_stats = _prototype_stats(
        feats=feats,
        protos=protos,
        assign=assign,
        sim=sim,
        frame_names=qualified_frame_names,
        proto_indices=proto_indices,
        selection_trace=selection_trace,
    )
    prototype_meta, instance_meta = build_proto_meta(
        source_name="shared",
        prototype_stats=prototype_stats,
        assign=assign,
        geom_guard=geom_guard,
        total_obs_count=int(len(qualified_frame_names)),
        obs_source_names=list(source_name_per_obs),
        all_source_names=list(flat["source_names"]),
        total_source_count=int(len(flat["source_names"])),
    )
    return PrototypeRepresentation(
        source_name="shared",
        prototype_features=protos,
        prototype_indices=proto_indices,
        prototype_frame_names=prototype_frame_names,
        proto_mass=proto_mass,
        frame_names=list(qualified_frame_names),
        feature_dim=int(protos.shape[1]),
        prototype_stats=prototype_stats,
        prototype_meta=prototype_meta,
        metadata={
            "prototype_stats_version": 2,
            "prototype_meta_version": 1,
            "builder_impl": "shared_source_normalized_cross_source_cover_gain",
            "input_frame_count": int(len(qualified_frame_names)),
            "selected_candidate_count": int(len(selection_trace)),
            "final_proto_count": int(len(proto_indices)),
            "source_names": list(flat["source_names"]),
            "shared_candidate_count": int(len(shared_candidates)),
            "prototype_config": {
                "kmin": int(config.prototype.kmin),
                "kmax": _effective_kmax(config.prototype),
                "gain_threshold": _effective_gain_threshold(config.prototype),
                "redundancy_lambda": float(config.prototype.redundancy_lambda),
                "cover_theta": float(config.prototype.cover_theta),
                "merge_similarity_threshold": float(config.prototype.merge_similarity_threshold),
                "prune_min_support": _effective_prune_min_support(config.prototype),
                "prune_min_ratio": float(config.prototype.prune_min_ratio),
                "geom_guard_gamma": float(config.prototype.geom_guard_gamma),
                "geom_guard_eta": float(config.prototype.geom_guard_eta),
            },
            "geom_guard_counts": {
                "normal": int(sum(1 for x in geom_guard if x.label == "normal")),
                "suspicious": int(sum(1 for x in geom_guard if x.label == "suspicious")),
                "bad": int(sum(1 for x in geom_guard if x.label == "bad")),
            },
            "instance_meta": {
                "proto_count": int(instance_meta.proto_count),
                "top1_support_ratio": float(instance_meta.top1_support_ratio),
                "cross_source_agreement_score": float(instance_meta.cross_source_agreement_score),
                "bad_geom_proto_ratio": float(instance_meta.bad_geom_proto_ratio),
                "metadata": dict(instance_meta.metadata),
            },
            "shared_candidates": [
                {
                    "global_index": int(item.global_index),
                    "source_name": str(item.source_name),
                    "source_local_index": int(item.source_local_index),
                    "frame_name": str(item.frame_name),
                    "frame_id": item.metadata.get("frame_id"),
                }
                for item in shared_candidates
            ],
        },
    )


def build_shared_instance_bundle(instance_pool: InstanceFeaturePool, config: FusionConfig) -> SourceRepresentationBundle:
    prototype_set = build_shared_prototypes(instance_pool, config)
    flat = _flatten_instance_sources(
        instance_pool,
        normalize_input=bool(config.normalize_input),
        geom_guard_gamma=float(config.prototype.geom_guard_gamma),
        geom_guard_eta=float(config.prototype.geom_guard_eta),
    )
    return SourceRepresentationBundle(
        source_name="shared",
        mean=build_shared_mean(instance_pool, config),
        prototype_set=prototype_set,
        geom_guard=list(flat["geom_guard"]),
        instance_meta=_instance_meta_from_repr(prototype_set),
        metadata={
            "builder_mode": "shared_multisource",
            "input_frame_count": int(flat["features"].shape[0]),
            "source_names": list(flat["source_names"]),
            "source_row_indices": {k: [int(x) for x in v] for k, v in dict(flat["source_row_indices"]).items()},
        },
    )


def build_source_bundle(source: FeatureSourcePool, config: FusionConfig) -> SourceRepresentationBundle:
    if bool(config.reliability.enabled):
        return build_reliability_guided_source_bundle(source, config)
    return build_plain_source_bundle(source, config)
