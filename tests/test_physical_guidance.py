"""CPU geometry and DDPM integration checks; no datasets or checkpoints needed."""

import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch


# Import the actual modules without networks/__init__.py, which eagerly imports
# optional rendering, CLIP and transformer packages unrelated to these tests.
PACKAGE = "diffuscene_guidance_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "scene_synthesis" / "networks")]
sys.modules[PACKAGE] = package
guidance = importlib.import_module(PACKAGE + ".physical_guidance")
ddpm = importlib.import_module(PACKAGE + ".diffusion_ddpm")


class PhysicalLossTests(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[[2., 2., 2., -0.4, 0., 0., 0.],
                                    [2., 2., 2., 0.6, 0., 0., 0.]]], dtype=torch.double)
        self.mask = torch.ones((1, 2), dtype=torch.bool)
        self.bounds = torch.tensor([[-3., -3., 3., 3.]], dtype=torch.double)

    def test_known_iou_and_separated_height(self):
        loss = guidance.compute_collision_guidance_loss(
            self.boxes, self.mask, iou_weight=1, penetration_weight=0, iou_threshold=0)
        self.assertAlmostEqual(float(loss), 1 / 3, places=7)
        self.boxes[0, 1, 4] = 3
        self.assertEqual(float(guidance.compute_collision_guidance_loss(self.boxes, self.mask)), 0)

    def test_rotated_rectangles_have_known_intersection(self):
        boxes = self.boxes.clone()
        boxes[..., [0, 2]] = torch.tensor([4., 2.], dtype=boxes.dtype)
        boxes[..., 3:6] = 0
        boxes[0, 1, 6] = math.pi / 2
        loss = guidance.compute_collision_guidance_loss(
            boxes, self.mask, iou_weight=1, penetration_weight=0, iou_threshold=0)
        # Two 4x2 rectangles cross at their centers: area=4, union area=12.
        self.assertAlmostEqual(float(loss), 1 / 3, places=7)

    def test_collision_gradient_reduces_overlap(self):
        boxes = self.boxes.clone().requires_grad_()
        loss = guidance.compute_collision_guidance_loss(boxes, self.mask)
        grad, = torch.autograd.grad(loss, boxes)
        self.assertGreater(float(grad[0, 0, 3]), 0)  # descent moves left object left
        self.assertLess(float(grad[0, 1, 3]), 0)
        updated = boxes.detach() - 0.01 * grad
        self.assertLess(float(guidance.compute_collision_guidance_loss(updated, self.mask)), float(loss))

    def test_obb_gradient_matches_finite_difference(self):
        boxes = self.boxes.clone()
        boxes[0, 0, 6], boxes[0, 1, 6] = 0.23, -0.41
        boxes[0, 1, 5] = 0.3
        boxes.requires_grad_()
        options = dict(iou_weight=1, penetration_weight=0, iou_threshold=0)
        loss = guidance.compute_collision_guidance_loss(boxes, self.mask, **options)
        grad, = torch.autograd.grad(loss, boxes)
        eps = 1e-5
        plus, minus = boxes.detach().clone(), boxes.detach().clone()
        plus[0, 1, 6] += eps
        minus[0, 1, 6] -= eps
        numerical = (guidance.compute_collision_guidance_loss(plus, self.mask, **options)
                     - guidance.compute_collision_guidance_loss(minus, self.mask, **options)) / (2 * eps)
        self.assertAlmostEqual(float(grad[0, 1, 6]), float(numerical), places=5)

    def test_parallel_identical_disjoint_and_empty_gradients_are_finite(self):
        for displacement in (0., 1., 2., 8.):
            for valid in (True, False):
                boxes = self.boxes.clone()
                boxes[0, 1, 3] = boxes[0, 0, 3] + displacement
                boxes.requires_grad_()
                mask = self.mask & valid
                loss = guidance.compute_collision_guidance_loss(boxes, mask)
                grad, = torch.autograd.grad(loss, boxes)
                self.assertTrue(torch.isfinite(grad).all())
                if not valid:
                    self.assertEqual(float(loss), 0)
                    self.assertEqual(float(grad.abs().sum()), 0)

    def test_single_object_collision_is_safe(self):
        boxes = self.boxes[:, :1].clone().requires_grad_()
        loss = guidance.compute_collision_guidance_loss(boxes, self.mask[:, :1])
        grad, = torch.autograd.grad(loss, boxes)
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(grad).all())

    def test_rotated_boundary_moves_object_inward(self):
        boxes = self.boxes[:, :1].clone()
        boxes[0, 0, :3] = torch.tensor([4., 2., 1.])
        boxes[0, 0, 3], boxes[0, 0, 6] = 2., math.pi / 4
        boxes.requires_grad_()
        loss = guidance.compute_room_boundary_loss(boxes, self.mask[:, :1], self.bounds)
        expected = 2 + 2.5 / math.sqrt(2) - 3
        self.assertAlmostEqual(float(loss), expected, places=7)
        grad, = torch.autograd.grad(loss, boxes)
        self.assertGreater(float(grad[0, 0, 3]), 0)
        updated = boxes.detach() - 0.01 * grad
        self.assertLess(float(guidance.compute_room_boundary_loss(updated, self.mask[:, :1], self.bounds)), float(loss))

    def test_walkable_modes_reduce_under_descent_and_mask_padding(self):
        for mode in ("center_penalty", "edge_gaussian"):
            boxes = self.boxes.clone().requires_grad_()
            loss = guidance.compute_walkability_loss(boxes, self.mask, self.bounds, type=mode)
            grad, = torch.autograd.grad(loss, boxes)
            self.assertTrue(torch.isfinite(grad).all())
            after = guidance.compute_walkability_loss(boxes.detach() - 0.01 * grad, self.mask, self.bounds, type=mode)
            self.assertLess(float(after), float(loss))
            zero = guidance.compute_walkability_loss(boxes, ~self.mask, self.bounds, type=mode)
            grad_zero, = torch.autograd.grad(zero, boxes)
            self.assertEqual(float(zero), 0)
            self.assertEqual(float(grad_zero.abs().sum()), 0)

    def test_losses_do_not_mix_scenes(self):
        boxes = self.boxes.expand(2, -1, -1).clone()
        boxes[1, :, 3] += 30
        mask = self.mask.expand(2, -1)
        bounds = self.bounds.expand(2, -1).clone()
        bounds[1, [0, 2]] += 30
        for function in (guidance.compute_room_boundary_loss, guidance.compute_walkability_loss):
            together = function(boxes, mask, bounds)
            separately = sum(function(boxes[b:b+1], mask[b:b+1], bounds[b:b+1]) for b in range(2)) / 2
            torch.testing.assert_close(together, separately)
        torch.testing.assert_close(guidance.compute_collision_guidance_loss(boxes, mask),
                                   guidance.compute_collision_guidance_loss(self.boxes, self.mask))


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stats_file = str(Path(self.temp.name) / "stats.json")
        with open(self.stats_file, "w") as handle:
            json.dump({"bounds_translations": [-5., -5., -5., 5., 5., 5.],
                       "bounds_sizes": [0., 0., 0., 2., 2., 2.],
                       "bounds_angles": [-math.pi, math.pi]}, handle)
        self.config = dict(angle_dim=2, class_dim=3, objectness_dim=0, objfeat_dim=2)
        self.guidance_config = dict(enabled=True, strength=2., start_ratio=0., interval=1,
                                    grad_clip=0., constraints={
                                        "collision": dict(weight=1.),
                                        "room_boundary": dict(weight=1.),
                                        "walkable": dict(weight=1.)})
        self.x0 = torch.tensor([[[-.08, 0., 0., -.5, -.5, -.5, 1., 0., 1., -1., -1., .2, .3],
                                 [.12, 0., 0., -.5, -.5, -.5, 1., 0., -1., 1., -1., .5, .7]]])

    def physical(self, cfg=None):
        return guidance.PhysicalGuidance(cfg or self.guidance_config, self.config, self.stats_file)

    def sampler(self, enabled=True, model_mean_type="x0", cfg=None):
        config = dict(self.config)
        config["inference_guidance"] = cfg or dict(self.guidance_config, enabled=enabled)
        return ddpm.GaussianDiffusion(config, np.linspace(.0001, .02, 10), "mse",
                                      model_mean_type, "fixedsmall", False, False, self.stats_file)

    def test_decode_half_sizes_order_and_cos_sin(self):
        boxes = self.physical().decode_boxes(self.x0)
        torch.testing.assert_close(boxes[0, 0, :3], torch.ones(3))
        self.assertAlmostEqual(float(boxes[0, 0, 3]), -.4, places=6)
        self.assertEqual(float(boxes[0, 0, 6]), 0)
        prediction = self.x0.clone()
        prediction[..., 6:8] = torch.tensor([0., 1.])
        torch.testing.assert_close(self.physical().decode_boxes(prediction)[..., 6],
                                   torch.full((1, 2), math.pi / 2))

    def test_mask_conventions(self):
        physical = self.physical()
        prediction = self.x0.clone()
        prediction[0, 1, 10] = 0
        self.assertEqual(physical.object_mask(prediction).tolist(), [[True, False]])
        config = dict(self.config, objectness_dim=1)
        physical = guidance.PhysicalGuidance(self.guidance_config, config, self.stats_file)
        prediction[0, :, 11] = torch.tensor([.2, -.2])
        self.assertEqual(physical.object_mask(prediction).tolist(), [[True, False]])

    def test_scalar_angle_and_batched_room_bounds(self):
        config = dict(self.config, angle_dim=1)
        physical = guidance.PhysicalGuidance(self.guidance_config, config, self.stats_file)
        prediction = self.x0[..., :7].expand(2, -1, -1).clone()
        prediction[..., 6] = .5  # normalized scalar angle -> pi/2
        boxes = physical.decode_boxes(prediction)
        torch.testing.assert_close(boxes[..., 6], torch.full((2, 2), math.pi / 2))
        bounds = torch.tensor([[-3., -3., 3., 3.], [-1., -2., 1., 2.]])
        grad = physical.gradient(prediction, 4, 10, bounds)
        self.assertEqual(grad.shape, prediction.shape)
        self.assertTrue(torch.isfinite(grad).all())

    def test_weights_can_independently_disable_every_loss(self):
        cfg = dict(self.guidance_config, constraints={name: dict(weight=0.) for name in self.guidance_config["constraints"]})
        self.assertIsNone(self.physical(cfg).gradient(self.x0, 4, 10))
        for name in self.guidance_config["constraints"]:
            cfg = dict(self.guidance_config, constraints={name: dict(weight=1.)})
            physical = self.physical(cfg)
            grad = physical.gradient(self.x0, 4, 10, [-.5, -.5, .5, .5])
            self.assertGreater(float(grad[..., :8].abs().sum()), 0)
            self.assertEqual(float(grad[..., 8:].abs().sum()), 0)
            self.assertIn(name, physical.step_stats[0])

    def test_per_constraint_schedule(self):
        cfg = dict(self.guidance_config, start_ratio=.95, interval=5,
                   constraints={"collision": dict(weight=1.),
                                "walkable": dict(weight=1., start_ratio=.5, interval=1)})
        physical = self.physical(cfg)
        collision_times = [t for t in reversed(range(1000)) if "collision" in physical.active_constraints(t, 1000)]
        self.assertEqual(collision_times, list(range(49, 0, -5)))
        self.assertEqual(set(physical.active_constraints(400, 1000)), {"walkable"})

    def test_missing_bounds_fails_clearly_but_collision_needs_no_bounds(self):
        with self.assertRaisesRegex(ValueError, "requires room_bounds"):
            self.physical().gradient(self.x0, 4, 10)
        cfg = dict(self.guidance_config, constraints={"collision": dict(weight=1.)})
        self.assertIsNotNone(self.physical(cfg).gradient(self.x0, 4, 10))

    def test_zero_losses_empty_slots_and_zero_angle_are_safe(self):
        prediction = self.x0.clone()
        prediction[..., 10] = 1  # every slot is empty
        prediction[..., 6:8] = 0
        for mode in ("center_penalty", "edge_gaussian"):
            cfg = dict(self.guidance_config, constraints=dict(self.guidance_config["constraints"],
                                                             walkable=dict(weight=1., type=mode)))
            grad = self.physical(cfg).gradient(prediction, 4, 10, [-3, -3, 3, 3])
            self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(float(grad.abs().sum()), 0)

    def test_guidance_runs_inside_no_grad_without_network_backprop(self):
        sampler = self.sampler()
        parameter = torch.nn.Parameter(self.x0.clone())
        def denoise(data, t, condition, condition_cross):
            return parameter.expand_as(data)
        data, t = self.x0.clone(), torch.tensor([4])
        with torch.no_grad():
            result = sampler.p_sample(denoise, data, t, None, None, torch.zeros,
                                      room_bounds=[-.5, -.5, .5, .5])
            baseline = self.sampler(enabled=False).p_sample(denoise, data, t, None, None, torch.zeros)
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(result.requires_grad)
        self.assertIsNone(parameter.grad)
        self.assertFalse(torch.equal(result[..., :8], baseline[..., :8]))
        torch.testing.assert_close(result[..., 8:], baseline[..., 8:], rtol=0, atol=0)

    def test_zero_weights_exactly_match_disabled_baseline(self):
        cfg = dict(self.guidance_config, constraints={name: dict(weight=0.) for name in self.guidance_config["constraints"]})
        disabled, zero_weights = self.sampler(enabled=False), self.sampler(cfg=cfg)
        data, t = self.x0.clone(), torch.tensor([4])
        denoise = lambda *args: self.x0
        torch.manual_seed(9)
        baseline = disabled.p_sample(denoise, data, t, None, None, torch.randn)
        torch.manual_seed(9)
        result = zero_weights.p_sample(denoise, data, t, None, None, torch.randn)
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)

    def test_all_prediction_types_and_t0(self):
        for model_type in ("eps", "x0", "v"):
            sampler = self.sampler(model_mean_type=model_type)
            for timestep in (4, 0):
                t = torch.tensor([timestep])
                result = sampler.p_sample(lambda *args: self.x0, self.x0, t, None, None, torch.zeros,
                                          room_bounds=[-3, -3, 3, 3])
                self.assertTrue(torch.isfinite(result).all())
                if timestep == 0:
                    baseline = self.sampler(enabled=False, model_mean_type=model_type).p_sample(
                        lambda *args: self.x0, self.x0, t, None, None, torch.zeros)
                    torch.testing.assert_close(result, baseline, rtol=0, atol=0)

    def test_gradient_clipping_is_per_object(self):
        physical = self.physical(dict(self.guidance_config, grad_clip=.01))
        grad = physical.gradient(self.x0, 4, 10, [-.5, -.5, .5, .5])
        self.assertTrue((grad.norm(dim=-1) <= .010001).all())

    def test_full_loop_and_trajectory_reset_logs(self):
        sampler = self.sampler()
        options = dict(denoise_fn=lambda *args: self.x0, shape=self.x0.shape,
                       device=torch.device("cpu"), condition=None, condition_cross=None,
                       noise_fn=torch.zeros, room_bounds=[-3, -3, 3, 3])
        result = sampler.p_sample_loop(**options)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(len(sampler.physical_guidance.step_stats), 10)
        trajectory = sampler.p_sample_loop_trajectory(freq=2, **options)
        self.assertEqual(len(sampler.physical_guidance.step_stats), 10)
        self.assertTrue(all(torch.isfinite(frame).all() for frame in trajectory))

    def test_completion_restores_known_boxes(self):
        sampler = self.sampler()
        result = sampler.p_sample_loop_complete(
            lambda *args: self.x0, self.x0.shape, torch.device("cpu"), None, None,
            noise_fn=torch.zeros, partial_boxes=self.x0[:, :1], room_bounds=[-3, -3, 3, 3])
        torch.testing.assert_close(result[:, :1], self.x0[:, :1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
