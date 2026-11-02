"""Independent checks for packed masks and benchmark aggregation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from msr3d.benchmark import aggregate, frame_metrics, verify_bundle


class BenchmarkTests(unittest.TestCase):
    def test_union_is_computed_before_pixel_iou(self):
        ground_truth = np.packbits([1, 1, 0, 0, 0, 0, 0, 0])
        masks = np.packbits([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]], axis=1)
        result = frame_metrics(ground_truth, ['a', 'b', 'c'], masks,
                               np.array([True, True, False]), ['a', 'b', 'c'])
        self.assertEqual(result['intersection'], 2)
        self.assertEqual(result['union'], 3)
        self.assertEqual(result['iou'], 2 / 3)
        self.assertEqual(result['missing_selected_ids'], ['c'])

    def test_unselected_masks_have_no_effect_and_empty_union_is_zero(self):
        self.assertEqual(frame_metrics(np.array([0], dtype=np.uint8), ['a'],
                         np.array([[255]], dtype=np.uint8), np.array([True]), [])['iou'], 0.)

    def test_unknown_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            frame_metrics(np.array([0], dtype=np.uint8), ['a'],
                          np.array([[0]], dtype=np.uint8), np.array([True]), ['b'])

    def test_scene_macro_gives_each_scene_equal_weight(self):
        rows = [dict(scene='a', miou=1., mf1=.8, macc=1),
                dict(scene='a', miou=1., mf1=.8, macc=1),
                dict(scene='b', miou=0., mf1=.2, macc=0)]
        result = aggregate(rows)
        self.assertEqual(result['query_macro']['miou'], 2 / 3)
        self.assertEqual(result['scene_macro']['miou'], .5)
        self.assertEqual(result['scene_macro']['mf1'], .5)

    def test_bundle_tamper_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'sample').write_bytes(b'input')
            metadata = {'format_version': 1, 'files': {
                'sample': {'size': 5, 'sha256': hashlib.sha256(b'input').hexdigest()}}}
            (root / 'manifest.json').write_text(json.dumps(metadata))
            self.assertEqual(verify_bundle(root), metadata)
            (root / 'sample').write_bytes(b'other')
            with self.assertRaises(ValueError):
                verify_bundle(root)
            metadata['files'] = {'../escape': {'size': 0, 'sha256': ''}}
            (root / 'manifest.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                verify_bundle(root)


if __name__ == '__main__':
    unittest.main()
