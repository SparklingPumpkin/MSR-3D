"""Fresh predictions can be evaluated without any matching feature cache."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from msr3d.benchmark import evaluate_predictions


class FreshEvaluationTests(unittest.TestCase):
    def test_evaluates_fresh_selection_without_opening_cached_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(
                root / "gt.npz",
                packed=np.packbits([1, 1, 0, 0, 0, 0, 0, 0]),
                shape=np.array([1, 8]),
            )
            np.savez(
                root / "masks.npz",
                instance_ids=np.array(["a", "b"]),
                packed=np.packbits(
                    [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]], axis=1
                ),
                present=np.array([True, True]),
                shape=np.array([1, 8]),
            )
            (root / "annotation.json").write_text(
                json.dumps(
                    {
                        "positive_ids": ["a"],
                        "frames": [
                            {
                                "frame": "view",
                                "instances": "masks.npz",
                                "ground_truth": "gt.npz",
                            }
                        ],
                    }
                )
            )
            query = {
                "dataset": "demo",
                "scene": "s",
                "query_id": "s::thing",
                "query_text": "thing",
                "splits": ["demo"],
                "annotation": "annotation.json",
                "features": "DOES_NOT_EXIST.npz",
            }
            prediction = {
                "dataset": "demo",
                "query_id": "s::thing",
                "selected_ids": ["a"],
            }
            with patch(
                "msr3d.benchmark.load_query",
                side_effect=AssertionError("Cached features were read"),
            ):
                result = evaluate_predictions(
                    root, [query], [prediction], metadata={"fresh": True}
                )
            self.assertEqual(result["summary"]["demo"]["scene_macro"]["miou"], 1.0)
            self.assertEqual(result["per_query"][0]["mf1"], 1.0)

    def test_rejects_predictions_for_a_different_query(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(
                Path("."),
                [{"dataset": "a", "query_id": "1"}],
                [{"dataset": "a", "query_id": "2"}],
            )


if __name__ == "__main__":
    unittest.main()
