from pathlib import Path
import tempfile
import unittest
import numpy as np
from msr3d.representation.scene import construct_scene, make_query_features


class RepresentationTest(unittest.TestCase):
    def test_source_balanced_prototypes_and_query_column_conventions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = np.zeros((3, 768), dtype=np.float32)
            features[0, 0] = features[1, 0] = features[2, 1] = 1.
            np.savez_compressed(root / 'pool.npz', features=features,
                instance_ids=np.asarray(['instance_0001'] * 3), sources=np.asarray(['render', 'sam', 'sam']),
                frame_names=np.asarray(['frame_00001', 'frame_00001', 'frame_00002']))
            construct_scene(root / 'pool.npz', root / 'rep')
            with np.load(root / 'rep/representation.npz', allow_pickle=False) as archive:
                self.assertEqual(int(archive['prototype_valid'].sum()), 2)
                np.testing.assert_allclose(archive['prototype_attributes'][0, :2, 0], [2/3, 1/3])
                np.testing.assert_allclose(archive['prototype_attributes'][0, :2, 1], [1., .5])
                self.assertAlmostEqual(float(archive['instance_attributes'][0, 2]), 5/6, places=6)
            query = np.zeros(768, dtype=np.float32)
            query[1] = 1
            item = make_query_features(root / 'rep', query, 'shared_pool_mean')
            np.testing.assert_allclose(item.matrix[0, :2, 0], [0., 1.])
            np.testing.assert_allclose(item.matrix[0, :2, 6:8], [[.5, .5], [1., 0.]])
            self.assertEqual(item.valid[0].tolist(), [True, True, False, False, False, False])
            self.assertEqual(item.anchor_instance_ids, ['instance_0001'])

    def test_duplicate_observation_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = np.zeros((2, 768), dtype=np.float32)
            features[:, 0] = 1
            np.savez_compressed(root / 'pool.npz', features=features, instance_ids=np.asarray(['i', 'i']),
                                sources=np.asarray(['sam', 'sam']), frame_names=np.asarray(['v', 'v']))
            with self.assertRaises(ValueError):
                construct_scene(root / 'pool.npz', root / 'rep')


if __name__ == '__main__':
    unittest.main()
