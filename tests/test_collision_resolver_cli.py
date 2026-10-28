"""CPU checks for the DiffuScene JSON-to-resolver adapter."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "run_resolve", Path(__file__).resolve().parents[1] / "scripts/eval/run_resolve.py")
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def renderer_overlap(data):
    """SAT using the mesh exporter's row-vector yaw, independently of solver."""
    footprints = []
    axes = []
    for size, center, angle in zip(data["sizes"][0], data["translations"][0], data["angles"][0]):
        c, s = math.cos(angle[0]), math.sin(angle[0])
        rotation = np.array([[c, -s], [s, c]])
        local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * np.array(size)[[0, 2]]
        footprints.append(local @ rotation + np.array(center)[[0, 2]])
        axes.extend(rotation)
    return all(min((footprints[0] @ a).max(), (footprints[1] @ a).max())
               - max((footprints[0] @ a).min(), (footprints[1] @ a).min()) > 1e-6
               for a in axes)


class ResolverAdapterTests(unittest.TestCase):
    def test_resolves_full_extent_and_renderer_yaw_preserving_metadata(self):
        examples = [
            ([[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [1.5, 0, 0]], [[0], [0]]),
            ([[2, .5, .2], [.2, .5, .2]], [[0, 0, 0], [1, 0, -1]], [[math.pi / 4], [0]]),
        ]
        for sizes, centers, angles in examples:
            with self.subTest(angles=angles), tempfile.TemporaryDirectory() as folder:
                data = dict(scene_ids=["Bedroom-1_0_000"], sizes=[sizes],
                            translations=[centers], angles=[angles],
                            class_labels=[[[1, 0], [0, 1]]], metadata="preserve")
                self.assertTrue(renderer_overlap(data))
                source, target = Path(folder) / "input.json", Path(folder) / "resolved.json"
                source.write_text(json.dumps(data), encoding="utf-8")
                resolver.resolve_json(source, target)
                result = json.loads(target.read_text(encoding="utf-8"))
                self.assertFalse(renderer_overlap(result))
                for key in data.keys() - {"translations"}:
                    self.assertEqual(result[key], data[key])
                self.assertEqual(json.loads(source.read_text(encoding="utf-8")), data)
                self.assertEqual([p[1] for p in result["translations"][0]], [0, 0])

    def test_duplicate_scene_ids_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "input.json", Path(folder) / "resolved.json"
            source.write_text(json.dumps(dict(scene_ids=["a", "a"])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                resolver.resolve_json(source, target)
            self.assertFalse(target.exists())

    def test_separated_heights_do_not_move(self):
        with tempfile.TemporaryDirectory() as folder:
            data = dict(scene_ids=["a"], sizes=[[[1, 1, 1], [1, 1, 1]]],
                        translations=[[[0, 0, 0], [0, 3, 0]]], angles=[[[0], [0]]])
            source, target = Path(folder) / "input.json", Path(folder) / "resolved.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            resolver.resolve_json(source, target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), data)


if __name__ == "__main__":
    unittest.main()
