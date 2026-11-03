import hashlib, json, tempfile, unittest
from pathlib import Path
from msr3d.inputs import validate_scene_assets


class SceneInputTests(unittest.TestCase):
    def bundle(self, root):
        scene = root / "preprocessed/lerf67/demo"
        for folder in ["images", "sparse", "rendered_instances"]:
            (scene / folder).mkdir(parents=True)
        files = {
            "scene.ply": b"ply",
            "instances.npy": b"ids",
            "config.yaml": b"cfg",
            "runtime.json": b"{}",
            "images/view.jpg": b"rgb",
            "sparse/cameras.bin": b"camera",
            "rendered_instances/instance_0001/view.png": b"render",
        }
        records = []
        for relative, value in files.items():
            p = scene / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(value)
            records.append(
                {
                    "path": p.relative_to(root).as_posix(),
                    "size": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
            )
        (root / "manifests").mkdir()
        (root / "manifests/assets.json").write_text(json.dumps({"files": records}))
        return scene

    def test_missing_render_source_is_rejected_before_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scene = self.bundle(root)
            import shutil

            shutil.rmtree(scene / "rendered_instances")
            with self.assertRaisesRegex(ValueError, "rendered_instances"):
                validate_scene_assets(root, "lerf67", "demo")

    def test_manifest_allows_sparse_views_but_rejects_missing_or_corrupt_asset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scene = self.bundle(root)
            self.assertEqual(validate_scene_assets(root, "lerf67", "demo")["files"], 7)
            p = scene / "rendered_instances/instance_0001/view.png"
            p.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_scene_assets(root, "lerf67", "demo")
            p.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                validate_scene_assets(root, "lerf67", "demo")


if __name__ == "__main__":
    unittest.main()
