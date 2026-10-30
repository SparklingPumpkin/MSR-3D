import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from msr3d import Matcher
from msr3d.features import QueryFeatures, load_query


def sample():
    rows = np.zeros((3, 6, 13), dtype=np.float32)
    rows[:, :2, 0] = [[.8, .6], [.5, .3], [.2, .1]]
    rows[:, :2, 1] = .5
    rows[:, :2, 6:8] = .5
    rows[:, :2, 8] = 2
    valid = np.zeros((3, 6), dtype=bool)
    valid[:, :2] = True
    return QueryFeatures(['object_a', 'object_b', 'object_c'], rows, valid,
                         ['object_a', 'object_b', 'object_c'], np.array([.8, .5, .2], dtype=np.float32))


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Matcher()

    def test_synthetic_prediction_and_protection(self):
        output = self.model.predict(sample())
        self.assertEqual(output['instance_ids'], sample().instance_ids)
        self.assertTrue(set(output['base_selected_ids']) <= set(output['selected_ids']))
        self.assertLessEqual(len(set(output['selected_ids']) - set(output['base_selected_ids'])), 1)
        self.assertTrue(np.isfinite(output['scores']).all())

    def test_repeated_forward_is_deterministic(self):
        self.assertEqual(self.model.predict(sample()), self.model.predict(sample()))

    def test_empty_valid_row_is_rejected(self):
        item = sample()
        item.valid[1] = False
        with self.assertRaises(ValueError):
            self.model.predict(item)

    def test_duplicate_ids_are_rejected(self):
        item = sample()
        item.instance_ids[1] = item.instance_ids[0]
        with self.assertRaises(ValueError):
            self.model.predict(item)

    def test_nonfinite_input_is_rejected(self):
        item = sample()
        item.matrix[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model.predict(item)

    def test_nonbinary_valid_is_rejected(self):
        item = sample()
        item.valid = item.valid.astype(float)
        item.valid[0, 0] = .5
        with self.assertRaises(ValueError):
            self.model.predict(item)

    def test_unsafe_identifier_is_rejected(self):
        item = sample()
        item.instance_ids[0] = '../object'
        with self.assertRaises(ValueError):
            self.model.predict(item)

    def test_missing_anchor_is_supported_and_input_is_unchanged(self):
        item = sample()
        item.anchor_instance_ids = item.anchor_instance_ids[:2]
        item.anchor_similarities = item.anchor_similarities[:2]
        before = item.matrix.copy()
        result = self.model.predict(item)
        self.assertEqual(result['instance_ids'], item.instance_ids)
        np.testing.assert_array_equal(before, item.matrix)

    def test_one_instance_and_constant_anchor_bank(self):
        item = sample()
        item.instance_ids = item.instance_ids[:1]
        item.matrix = item.matrix[:1]
        item.valid = item.valid[:1]
        item.anchor_similarities[:] = .4
        result = self.model.predict(item)
        self.assertEqual(len(result['scores']), 1)
        self.assertTrue(np.isfinite(result['scores']).all())

    def test_npz_round_trip_without_pickle(self):
        item = sample()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.npz'
            np.savez(path, instance_ids=np.asarray(item.instance_ids), matrix=item.matrix,
                     valid=item.valid, anchor_instance_ids=np.asarray(item.anchor_instance_ids),
                     anchor_similarities=item.anchor_similarities)
            self.assertEqual(self.model.predict(item), self.model.predict(load_query(path)))

    def test_invalid_weight_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'matcher.json').write_text(json.dumps({'format_version': 1, 'weights_sha256': '0' * 64}))
            (root / 'matcher.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                Matcher(root)


if __name__ == '__main__':
    unittest.main()
