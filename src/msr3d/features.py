"""Validated cached feature interface; no image encoder or dataset dependency."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def _identifiers(values, name):
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f'{name} must be a sequence of strings')
    result = list(values)
    if not result or any(not isinstance(x, str) or not x or '/' in x or '\\' in x or any(ord(c) < 32 for c in x) for x in result):
        raise ValueError(f'{name} must contain nonempty identifiers without path separators or control characters')
    if len(set(result)) != len(result):
        raise ValueError(f'{name} must be unique')
    return result


@dataclass
class QueryFeatures:
    instance_ids: list[str]
    matrix: np.ndarray
    valid: np.ndarray
    anchor_instance_ids: list[str]
    anchor_similarities: np.ndarray

    def validated(self):
        ids = _identifiers(self.instance_ids, 'instance_ids')
        anchor_ids = _identifiers(self.anchor_instance_ids, 'anchor_instance_ids')
        matrix = np.array(self.matrix, dtype=np.float32, copy=True)
        valid = np.asarray(self.valid)
        anchor = np.array(self.anchor_similarities, dtype=np.float32, copy=True)
        if matrix.shape != (len(ids), 6, 13):
            raise ValueError('matrix must have shape (number of instances, 6, 13)')
        if valid.shape != matrix.shape[:2] or not np.isin(valid, [0, 1]).all():
            raise ValueError('valid must be a binary array of shape (number of instances, 6)')
        valid = valid.astype(bool, copy=True)
        if not valid.any(axis=1).all():
            raise ValueError('each instance needs at least one valid prototype')
        if anchor.shape != (len(anchor_ids),):
            raise ValueError('anchor_similarities must match anchor_instance_ids')
        if not np.isfinite(matrix).all() or not np.isfinite(anchor).all():
            raise ValueError('features must be finite')
        return QueryFeatures(ids, matrix, valid, anchor_ids, anchor)


def load_query(path: str | Path) -> QueryFeatures:
    """Read one query from an NPZ containing only the documented five arrays."""
    fields = ('instance_ids', 'matrix', 'valid', 'anchor_instance_ids', 'anchor_similarities')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise ValueError('input must contain exactly: ' + ', '.join(fields))
        item = QueryFeatures(archive['instance_ids'].tolist(), archive['matrix'].copy(),
                             archive['valid'].copy(), archive['anchor_instance_ids'].tolist(),
                             archive['anchor_similarities'].copy())
    return item.validated()
