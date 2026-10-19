"""Regression checks for inference guidance JSONL logs."""

import io
import json
import unittest

import numpy as np
import torch

from scene_synthesis.serialization import json_default


class GuidanceLogSerializationTests(unittest.TestCase):
    def test_guidance_record_with_numpy_metadata_round_trips_as_jsonl(self):
        record = {
            "scene_id": np.array("LivingDiningRoom-2065"),
            "scene_index": np.int64(0),
            "room_bounds": np.array([-3., -2., 3., 2.]),
            "config": {"constraints": {"collision": {"weight": np.float32(10)}}},
            "step_stats": [{"timestep": np.int32(49), "collision": np.float64(.25),
                            "active": np.bool_(True)}],
        }
        log = io.StringIO()
        log.write(json.dumps(record, default=json_default) + "\n")
        log.write(json.dumps(record, default=json_default) + "\n")
        lines = log.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        restored = json.loads(lines[0])
        self.assertEqual(restored["scene_id"], "LivingDiningRoom-2065")
        self.assertEqual(restored["room_bounds"], [-3., -2., 3., 2.])
        self.assertEqual(restored["config"]["constraints"]["collision"]["weight"], 10.)
        self.assertEqual(restored["step_stats"][0], {"timestep": 49, "collision": .25, "active": True})

    def test_tensor_metadata_detaches_without_changing_numeric_structure(self):
        value = torch.tensor([[1., 2.]], requires_grad=True)
        restored = json.loads(json.dumps({"gradient": value}, default=json_default))
        self.assertEqual(restored, {"gradient": [[1., 2.]]})

    def test_unsupported_objects_still_raise(self):
        with self.assertRaisesRegex(TypeError, "not JSON serializable"):
            json.dumps({"unsupported": object()}, default=json_default)


if __name__ == "__main__":
    unittest.main()
