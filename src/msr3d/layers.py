"""Scalar arithmetic for prototype scoring and query calibration."""
from __future__ import annotations
from typing import Dict, List, Tuple, Sequence
import math
import torch
from torch import nn

class PrototypeScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, row_features: torch.Tensor) -> torch.Tensor:
        return self.linear(row_features).squeeze(-1)

class PrototypeAggregator(nn.Module):
    def __init__(self, instance_meta_dim: int, topk_for_topmean: int = 2) -> None:
        super().__init__()
        self.topk_for_topmean = max(1, int(topk_for_topmean))
        self.row_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.instance_meta_linear = nn.Linear(instance_meta_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, row_scores: torch.Tensor, instance_meta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if row_scores.ndim != 1 or int(row_scores.numel()) == 0:
            raise ValueError("row_scores must be a non-empty 1D tensor")
        if instance_meta.ndim != 1:
            raise ValueError("instance_meta must be a 1D tensor")
        r_max = torch.max(row_scores)
        topk = min(self.topk_for_topmean, int(row_scores.numel()))
        r_topk_mean = torch.topk(row_scores, k=topk).values.mean()
        r_mean = row_scores.mean()
        row_part = (
            self.row_weights[0] * r_max
            + self.row_weights[1] * r_topk_mean
            + self.row_weights[2] * r_mean
        )
        inst_part = self.instance_meta_linear(instance_meta.view(1, -1)).squeeze(0).squeeze(0)
        score = row_part + inst_part + self.bias.squeeze(0)
        return score, {
            "r_max": float(r_max.item()),
            "r_topk_mean": float(r_topk_mean.item()),
            "r_mean": float(r_mean.item()),
            "instance_bias_term": float(inst_part.item()),
        }

def _safe_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    vec = values.view(-1).float()
    if int(vec.numel()) == 0:
        return torch.zeros((k,), dtype=torch.float32, device=vec.device)
    topk = torch.topk(vec, k=min(k, int(vec.numel()))).values
    if int(topk.numel()) < k:
        topk = torch.cat([topk, topk.new_zeros(k - int(topk.numel()))], dim=0)
    return topk

def _safe_entropy(values: torch.Tensor) -> torch.Tensor:
    vec = values.view(-1).float().clamp_min(1e-8)
    if int(vec.numel()) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=values.device)
    probs = vec / vec.sum().clamp_min(1e-8)
    return -(probs * probs.log()).sum()

def build_query_summary_features(final_scores: torch.Tensor) -> torch.Tensor:
    score_vec = final_scores.view(-1).float().clamp(0.0, 1.0)
    topk = _safe_topk(score_vec, 5)
    count = float(int(score_vec.numel()))
    mean = score_vec.mean() if int(score_vec.numel()) > 0 else score_vec.new_tensor(0.0)
    std = score_vec.std(unbiased=False) if int(score_vec.numel()) > 1 else score_vec.new_tensor(0.0)
    spread = score_vec.max() - score_vec.min() if int(score_vec.numel()) > 0 else score_vec.new_tensor(0.0)
    high_08 = float((score_vec > 0.8).sum().item())
    high_09 = float((score_vec > 0.9).sum().item())
    high_095 = float((score_vec > 0.95).sum().item())
    gap12 = float(topk[0] - topk[1]) if count >= 2 else float(topk[0])
    gap13 = float(topk[0] - topk[2]) if count >= 3 else float(topk[0])
    gap15 = float(topk[0] - topk[4]) if count >= 5 else float(topk[0])
    entropy = _safe_entropy(score_vec)
    tail_mean = torch.topk(score_vec, k=min(3, int(score_vec.numel()))).values.mean() if int(score_vec.numel()) > 0 else score_vec.new_tensor(0.0)
    feature_list = [
        float(topk[0]),
        float(topk[1]),
        float(topk[2]),
        float(topk[4]),
        gap12,
        gap13,
        gap15,
        high_08,
        high_09,
        high_095,
        float(mean),
        float(std),
        float(spread),
        float(entropy),
        count,
        float(tail_mean),
    ]
    return torch.tensor(feature_list, dtype=torch.float32, device=score_vec.device)

def sorted_percentile(sorted_values: Sequence[float], fraction: float) -> float:
    values = list(sorted_values)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    fraction = min(max(float(fraction), 0.0), 1.0)
    idx = (len(values) - 1) * fraction
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)

def robust_scale_stats(
    values: Sequence[float],
    *,
    lower_quantile: float,
    upper_quantile: float,
    eps: float,
    mad_scale_factor: float,
) -> Dict[str, object]:
    sorted_values = sorted(float(v) for v in values)
    if not sorted_values:
        return {
            "scale": 1.0,
            "scale_source": "unit_fallback",
            "q_lower": 0.0,
            "q_upper": 0.0,
            "median": 0.0,
            "mad_scale": 0.0,
            "absmax": 0.0,
            "count": 0,
        }
    q_lower = sorted_percentile(sorted_values, lower_quantile)
    q_upper = sorted_percentile(sorted_values, upper_quantile)
    q_span = float(q_upper - q_lower)
    median = sorted_percentile(sorted_values, 0.5)
    abs_dev = sorted(abs(v - median) for v in sorted_values)
    mad_scale = float(mad_scale_factor * sorted_percentile(abs_dev, 0.5))
    absmax = float(max(abs(v) for v in sorted_values))
    if q_span > eps:
        scale = q_span
        scale_source = "qspan"
    elif mad_scale > eps:
        scale = mad_scale
        scale_source = "mad"
    elif absmax > eps:
        scale = absmax
        scale_source = "absmax"
    else:
        scale = 1.0
        scale_source = "unit_fallback"
    return {
        "scale": float(scale),
        "scale_source": scale_source,
        "q_lower": float(q_lower),
        "q_upper": float(q_upper),
        "median": float(median),
        "mad_scale": float(mad_scale),
        "absmax": float(absmax),
        "count": len(sorted_values),
    }
