"""Evaluate portable preprocessed benchmark features with the bundled matcher."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import torch

from . import Matcher, load_query, __version__


def _path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Invalid bundle path: {relative}')
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Bundle path escapes root: {relative}')
    return target


def _sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_bundle(root):
    """Validate the complete asset inventory before evaluating a data bundle."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['format_version'] != 1:
        raise ValueError('Unsupported data format')
    for relative, expected in manifest['files'].items():
        path = _path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or linked asset: {relative}')
        if path.stat().st_size != expected['size'] or _sha256(path) != expected['sha256']:
            raise ValueError(f'Asset checksum mismatch: {relative}')
    return manifest


def frame_metrics(ground_truth, instance_ids, masks, present, selected_ids):
    """Union processed instance masks, then count pixel intersection and union."""
    if masks.dtype != np.uint8 or masks.shape != (len(instance_ids), ground_truth.size):
        raise ValueError('Inconsistent packed mask bank')
    if len(set(instance_ids)) != len(instance_ids) or present.shape != (len(instance_ids),):
        raise ValueError('Invalid mask inventory')
    lookup = {iid: index for index, iid in enumerate(instance_ids)}
    union = np.zeros_like(ground_truth)
    missing = []
    for iid in sorted(set(selected_ids)):
        if iid not in lookup:
            raise ValueError(f'Prediction absent from candidate mask inventory: {iid}')
        index = lookup[iid]
        if present[index]:
            union |= masks[index]
        else:
            missing.append(iid)
    intersection = int(np.bitwise_count(union & ground_truth).sum())
    denominator = int(np.bitwise_count(union | ground_truth).sum())
    return {'intersection': intersection, 'union': denominator,
            'iou': intersection / denominator if denominator else 0.,
            'missing_selected_mask_count': len(missing), 'missing_selected_ids': missing}


def aggregate(rows):
    def means(records):
        return {key: float(np.mean([r[key] for r in records])) for key in ('miou', 'mf1', 'macc')}
    if not rows:
        raise ValueError('Cannot aggregate an empty cohort')
    scenes = {scene: {'query_count': sum(r['scene'] == scene for r in rows),
                      **means([r for r in rows if r['scene'] == scene])}
              for scene in sorted({r['scene'] for r in rows})}
    return {'query_count': len(rows), 'scene_count': len(scenes), 'query_macro': means(rows),
            'scene_macro': means(list(scenes.values())), 'scenes': scenes}


def evaluate(root, split='all'):
    """Run inference first; read annotations only for subsequent metric computation."""
    root = Path(root)
    manifest = verify_bundle(root)
    for kind, filename in (('weights', 'matcher.npz'), ('config', 'matcher.json')):
        if _sha256(Path(__file__).parent / 'assets' / filename) != manifest[f'model_{kind}_sha256']:
            raise ValueError(f'Data references a different model {kind} version')
    queries = json.loads(_path(root, manifest['queries']).read_text())
    if len({(q['dataset'], q['query_id']) for q in queries}) != len(queries):
        raise ValueError('Duplicate queries in data bundle')
    if split != 'all':
        queries = [q for q in queries if split in q['splits']]
    if not queries:
        raise ValueError('No queries in requested split')
    torch.set_num_threads(1)
    model = Matcher()
    predictions = []
    for query in queries:
        result = model.predict(load_query(_path(root, query['features'])))
        predictions.append({'dataset': query['dataset'], 'query_id': query['query_id'], **result})
    per_query, per_frame = [], []
    # Each scene/frame/shape has a single immutable mask bank, reused across queries.
    cache_path, cache = None, None
    for query, prediction in zip(queries, predictions):
        annotation = json.loads(_path(root, query['annotation']).read_text())
        selected = set(prediction['selected_ids'])
        positive = set(annotation['positive_ids'])
        frames = []
        for frame in annotation['frames']:
            if frame['instances'] != cache_path:
                with np.load(_path(root, frame['instances']), allow_pickle=False) as archive:
                    cache = {key: archive[key] for key in archive.files}
                cache_path = frame['instances']
            with np.load(_path(root, frame['ground_truth']), allow_pickle=False) as archive:
                ground_truth, shape = archive['packed'], archive['shape']
            if not np.array_equal(shape, cache['shape']):
                raise ValueError('GT and render dimensions differ')
            packed_size = (int(np.prod(shape)) + 7) // 8
            if ground_truth.dtype != np.uint8 or ground_truth.shape != (packed_size,):
                raise ValueError('Invalid packed GT dimensions')
            row = {'dataset': query['dataset'], 'query_id': query['query_id'],
                   'scene': query['scene'], 'frame': frame['frame'],
                   **frame_metrics(ground_truth, cache['instance_ids'].tolist(),
                                   cache['packed'], cache['present'], prediction['selected_ids'])}
            frames.append(row)
            per_frame.append(row)
        if not frames:
            raise ValueError('Query has no annotated evaluation frames')
        miou = float(np.mean([r['iou'] for r in frames]))
        per_query.append({'dataset': query['dataset'], 'query_id': query['query_id'],
                          'scene': query['scene'], 'query_text': query['query_text'],
                          'splits': query['splits'], 'selected_ids': sorted(selected),
                          'positive_ids': sorted(positive), 'tp_ids': sorted(positive & selected),
                          'fp_ids': sorted(selected - positive), 'fn_ids': sorted(positive - selected),
                          'frame_count': len(frames), 'miou': miou,
                          'mf1': 2 * len(selected & positive) / (len(selected) + len(positive))
                                 if selected or positive else 0., 'macc': int(miou > .25)})
    splits = sorted({s for row in per_query for s in row['splits']})
    summary = {s: aggregate([r for r in per_query if s in r['splits']]) for s in splits
               if split == 'all' or s == split}
    metadata = {'data_version': manifest['data_version'], 'package_version': __version__,
                'manifest_sha256': _sha256(root / 'manifest.json'),
                'model_weights_sha256': manifest['model_weights_sha256'],
                'model_config_sha256': manifest['model_config_sha256'], 'requested_split': split,
                'environment': {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__}}
    return {'data_version': manifest['data_version'], 'metadata': metadata, 'predictions': predictions,
            'per_query': per_query, 'per_frame': per_frame, 'summary': summary}


def check_reference(result, root):
    """Compare saved references only after all fresh inference and evaluation."""
    reference = json.loads((Path(root) / 'reference/results.json').read_text())
    for field in ('predictions', 'per_query', 'per_frame'):
        keys = ('dataset', 'query_id', 'frame') if field == 'per_frame' else ('dataset', 'query_id')
        lookup = {tuple(row[k] for k in keys): row for row in reference[field]}
        for row in result[field]:
            expected = lookup[tuple(row[k] for k in keys)]
            if row != expected:
                raise ValueError(f'Reference mismatch: {field} {tuple(row[k] for k in keys)}')
    for split, summary in result['summary'].items():
        if summary != reference['summary'][split]:
            raise ValueError(f'Reference summary mismatch: {split}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=('all', 'lerf67', 'lerf45', '3dovs61'), default='all')
    parser.add_argument('--check-reference', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    result = evaluate(args.data_root, args.split)
    if args.check_reference:
        check_reference(result, args.data_root)
    args.output.mkdir(parents=True, exist_ok=False)
    for field in ('metadata', 'predictions', 'per_query', 'per_frame', 'summary'):
        (args.output / f'{field}.json').write_text(json.dumps(result[field], indent=2) + '\n')
    for field in ('per_query', 'per_frame'):
        rows = result[field]
        with (args.output / f'{field}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
                             for row in rows)
    for split, summary in result['summary'].items():
        scene = summary['scene_macro']
        print(f"{split}: {summary['query_count']} queries, {summary['scene_count']} scenes; "
              f"scene-macro mIoU={100 * scene['miou']:.4f}%, mF1={100 * scene['mf1']:.4f}%")
    if args.check_reference:
        print('All predictions and per-frame, per-query and aggregate metrics match references exactly.')


if __name__ == '__main__':
    main()
