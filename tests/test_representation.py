from pathlib import Path
import tempfile
import unittest
import numpy as np
from msr3d.representation.scene import construct_scene, make_query_features


class RepresentationTest(unittest.TestCase):
    def test_geometric_query_uses_independent_valid_normalized_anchor_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prototypes = np.zeros((2, 6, 768), dtype=np.float32)
            prototypes[0, 0, 0] = prototypes[0, 3, 1] = prototypes[1, 2, 0] = 1.
            valid = np.zeros((2, 6), dtype=bool)
            valid[0, [0, 3]] = valid[1, 2] = True
            np.savez(root / 'representation.npz',
                instance_ids=np.asarray(['instance_0001', 'instance_0002']),
                prototype_features=prototypes, prototype_valid=valid,
                prototype_attributes=np.zeros((2, 6, 7), dtype=np.float32),
                instance_attributes=np.zeros((2, 4), dtype=np.float32),
                shared_means=np.zeros((2, 768), dtype=np.float32))
            anchors = np.zeros((3, 768), dtype=np.float32)
            anchors[:, :2] = [[0., -9.], [4., 0.], [3., 4.]]
            np.savez(root / 'anchors.npz',
                instance_ids=np.asarray(['instance_0099', 'instance_0001', 'instance_0002']),
                embeddings=anchors, valid=np.asarray([True, False, True]))
            query = np.zeros(768, dtype=np.float32)
            query[:2] = [6., 8.]

            item = make_query_features(root, query, 'geometric_topk', root / 'anchors.npz')

            self.assertEqual(item.instance_ids, ['instance_0001', 'instance_0002'])
            self.assertEqual(item.anchor_instance_ids, ['instance_0099', 'instance_0002'])
            np.testing.assert_allclose(item.anchor_similarities, [-.8, 1.], atol=1e-6)
            np.testing.assert_array_equal(item.valid, valid)
            np.testing.assert_allclose(item.matrix[:, :, 0][valid], [.6, .8, .6], atol=1e-6)
            np.testing.assert_array_equal(item.matrix[~valid], 0.)

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
