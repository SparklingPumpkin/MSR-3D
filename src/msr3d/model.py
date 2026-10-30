"""MSR-3D matching from precomputed prototype evidence, on CPU."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from scipy.special import expit, logit, softmax
from scipy.stats import rankdata
from .features import QueryFeatures
from .layers import PrototypeScorer, PrototypeAggregator, build_query_summary_features, robust_scale_stats


def _scale(values):
    return robust_scale_stats(list(map(float, values)), lower_quantile=.1,
                              upper_quantile=.9, eps=1e-6, mad_scale_factor=1.4826)['scale']


class Matcher:
    """Fixed-weight matcher. Predictions do not read labels or reference results."""

    def __init__(self, weights_dir: str | Path | None = None):
        directory = Path(weights_dir) if weights_dir is not None else Path(__file__).parent / 'assets'
        self.config = json.loads((directory / 'matcher.json').read_text())
        if self.config.get('format_version') != 1:
            raise ValueError('unsupported weight format')
        weights = directory / 'matcher.npz'
        if hashlib.sha256(weights.read_bytes()).hexdigest() != self.config['weights_sha256']:
            raise ValueError('weight checksum mismatch')
        with np.load(weights, allow_pickle=False) as archive:
            self.weights = {k: archive[k].copy() for k in archive.files}
        cfg = self.config['semantic']
        self.row = PrototypeScorer(8)
        self.aggregator = PrototypeAggregator(4, cfg['topk'])
        self.threshold = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        self.boundary = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 1))
        for module, prefix in ((self.row, 'semantic/backbone.row_scorer.'),
                               (self.aggregator, 'semantic/backbone.aggregator.'),
                               (self.threshold, 'semantic/tau_net.net.'),
                               (self.boundary, 'boundary/residual.')):
            state = {key[len(prefix):]: torch.from_numpy(value) for key, value in self.weights.items() if key.startswith(prefix)}
            module.load_state_dict(state)
            module.eval()
            module.requires_grad_(False)

    def _semantic_scores(self, item):
        scores = []
        for matrix, valid in zip(torch.from_numpy(item.matrix), torch.from_numpy(item.valid)):
            rows = matrix[valid]
            scores.append(self.aggregator(self.row(rows[:, :8]), rows[0, 8:12])[0].view(()))
        raw = torch.stack(scores).view(-1).float()
        cfg = self.config['semantic']
        values = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-6)).pow(cfg['score_power'])
        threshold = cfg['threshold_min'] + (cfg['threshold_max'] - cfg['threshold_min']) * torch.sigmoid(
            self.threshold(build_query_summary_features(values).view(1, -1)).squeeze(0).squeeze(0))
        values = values.numpy().astype(float)
        margin = values - float(threshold)
        semantic = np.tanh(margin / max(_scale(margin), 1e-6))
        anchor_raw = torch.from_numpy(item.anchor_similarities)
        shifted = anchor_raw - anchor_raw.min()
        if float(shifted.max()) <= 1e-12:
            relative = torch.zeros_like(shifted)
            relative[int(anchor_raw.argmax())] = 1.
        else:
            relative = shifted / shifted.max()
        index = {iid: i for i, iid in enumerate(item.anchor_instance_ids)}
        common = np.array([float(relative[index[iid]]) if iid in index else 0. for iid in item.instance_ids])
        full = list(map(float, relative.tolist()))
        full += [0. for iid in item.instance_ids if iid not in index]
        anchor = np.tanh((common - self.config['anchor_threshold']) / max(
            _scale(np.asarray(full) - self.config['anchor_threshold']), 1e-6))
        selected = {iid for iid, score in zip(item.instance_ids, values) if score >= float(threshold)}
        order = sorted((i for i, iid in enumerate(item.instance_ids) if iid not in selected),
                       key=lambda i: (-values[i], item.instance_ids[i]))
        return semantic, anchor, [item.instance_ids[i] for i in order[:3]]

    def _base_margin(self, ids, semantic, anchor, proposals):
        first = torch.tensor(semantic, dtype=torch.float32)
        second = torch.tensor(anchor, dtype=torch.float32)
        initial = torch.maximum(first, second)
        margin = initial + self.boundary(torch.stack((first, second, first * second,
            first + second, first - second, initial), dim=-1)).squeeze(-1)
        core = [iid for iid, keep in zip(ids, margin > 0.) if keep]
        if not core:
            margin = initial
            core = [iid for iid, keep in zip(ids, initial > 0.) if keep]
        candidates = [(iid, i) for i, iid in enumerate(ids) if iid in set(proposals) - set(core)]
        candidates.sort()
        values = np.empty(0)
        if candidates:
            x = np.array([[semantic[i], anchor[i]] for _, i in candidates], dtype=float)
            x = (x - self.weights['supplementary/mean']) / self.weights['supplementary/scale']
            for layer in range(2):
                x = x @ self.weights[f'supplementary/w{layer}'] + self.weights[f'supplementary/b{layer}']
                x = expit(x) if layer == 1 else np.maximum(x, 0.)
            values = x[:, 0]
        ranked = sorted(((float(value), rank, iid) for rank, ((iid, _), value) in enumerate(zip(candidates, values))),
                        key=lambda row: (-row[0], row[1], row[2]))
        policy = self.config['supplementary']
        adds = [iid for value, _, iid in ranked if value >= policy['threshold']] if policy['enabled'] else []
        if policy['max_add'] is not None:
            adds = adds[:policy['max_add']]
        selected = set(core + adds)
        result = np.tanh(margin.numpy().astype(float))
        candidates = ranked if policy['max_add'] is None else ranked[:policy['max_add']]
        index = {iid: i for i, iid in enumerate(ids)}
        if policy['enabled']:
            threshold = logit(np.clip(policy['threshold'], 1e-12, 1 - 1e-12))
            for value, _, iid in candidates:
                position = index[iid]
                candidate_margin = np.tanh(logit(np.clip(value, 1e-12, 1 - 1e-12)) - threshold)
                result[position] = max(result[position], candidate_margin)
        positive = np.array([iid in selected for iid in ids])
        repaired = (result > 0.) != positive
        result[repaired] = np.where(positive[repaired], 1e-6, -1e-6)
        result[(result == 0.) & ~positive] = -1e-6
        return result

    def _ambiguity(self, item, semantic, anchor):
        matrix = item.matrix.astype(float)
        valid = item.valid
        matrix = matrix.copy()
        matrix[~valid] = 0.
        similarities = matrix[:, :, 0]
        statistics = []
        for values, mask in zip(similarities, valid):
            values = values[mask]
            ordered = np.sort(values)[::-1]
            statistics.append([ordered[0], ordered[:2].mean(), values.mean(), ordered[-1], values.std()])
        statistics = np.asarray(statistics)
        ranks = np.ones_like(statistics) if len(statistics) == 1 else 1 - (rankdata(-statistics, axis=0, method='average') - 1) / (len(statistics) - 1)
        summary = np.column_stack((statistics, ranks, statistics.max(axis=0) - statistics,
                                  semantic, anchor, np.tanh(semantic), np.tanh(anchor), semantic * anchor))
        weights = self.weights
        prefix = 'ambiguity/'
        mean = weights[prefix + 'semantic_scaler/mean']
        std = weights[prefix + 'semantic_scaler/std']
        active = weights[prefix + 'semantic_scaler/active']
        summary = ((summary - mean) / std)[:, active]
        mean = weights[prefix + 'row_scaler/mean']
        std = weights[prefix + 'row_scaler/std']
        active = weights[prefix + 'row_scaler/active']
        centered = (similarities - np.sum(similarities * valid, axis=1, keepdims=True) / valid.sum(axis=1, keepdims=True)) / std[0]
        quality = (matrix[:, :, (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)] - mean) / std
        prototype = quality[:, :, 1:7][:, :, active[1:7]]
        instance = quality[:, :, 7:10][:, :, active[7:10]]
        rows = np.concatenate((centered[:, :, None], prototype, prototype * centered[:, :, None], instance * centered[:, :, None]), axis=2)
        rows[~valid] = 0.
        coefficients = weights[prefix + 'coefficients']
        width = summary.shape[1]
        value = summary @ coefficients[:width]
        logits = np.einsum('ird,d->ir', rows, coefficients[width:])
        logits = np.where(valid, 2 * np.tanh(logits), -1e300)
        attention = softmax(logits, axis=1)
        weighted = (attention * similarities).sum(axis=1)
        uniform = (similarities * softmax(np.where(valid, 0., -1e300), axis=1)).sum(axis=1)
        value = value + (weighted - uniform) / std[0]
        return 2. * np.tanh(value)

    @torch.no_grad()
    def predict(self, features: QueryFeatures) -> dict:
        """Return instance-aligned scores and the final protected instance set."""
        item = features.validated()
        semantic, anchor, proposals = self._semantic_scores(item)
        base = self._base_margin(item.instance_ids, semantic, anchor, proposals)
        correction = self._ambiguity(item, semantic, anchor)
        corrected = base + correction
        margin = base.copy()
        eligible = np.flatnonzero((base < 0.) & (corrected > 0.))
        if len(eligible):
            winner = min(eligible, key=lambda i: (-corrected[i], item.instance_ids[i]))
            margin[winner] = corrected[winner]
        scores = expit(4 * margin)
        if not all(np.isfinite(x).all() for x in (base, correction, corrected, scores)):
            raise ValueError('nonfinite numerical output')
        selected = sorted(iid for iid, score in zip(item.instance_ids, scores) if score > self.config['selection_threshold'])
        return {'instance_ids': item.instance_ids, 'selected_ids': selected,
                'base_selected_ids': sorted(iid for iid, value in zip(item.instance_ids, base) if value > 0.),
                'scores': scores.tolist(), 'ambiguity_scores': correction.tolist(),
                'base_margins': base.tolist(), 'corrected_margins': corrected.tolist()}
