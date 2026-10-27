# Physical guidance during inference

The standalone implementation is in `scene_synthesis/networks/physical_guidance.py`.
It does not import anything from `echoscene`. Existing checkpoints work without
retraining. Guidance is disabled unless `network.inference_guidance.enabled` is
true or the generation script is given the optional guidance configuration.

## Run and tune

Keep the existing generation command and append:

```text
--guidance_config config/inference_guidance.yaml --seed 111
```

The config is a separate file: the positional model YAML still determines the
architecture, eps/x0/v prediction type, dataset paths, and checkpoint settings.
The guidance YAML overrides the whole `network.inference_guidance` mapping.
It can also be pasted under `network.inference_guidance` in the model YAML.

Tune these three independent multipliers:

```yaml
constraints:
  collision:
    weight: 10.0
  room_boundary:
    weight: 10.0
  walkable:
    weight: 1.0
```

This snippet illustrates the keys; the supplied YAML includes the full settings.
Set a weight to `0` (or `enabled: false`) to disable that loss. Set the top-level
`enabled: false` for the baseline. Use identical scene selection, seed, start index,
sequence count and model checkpoint for comparisons, with a separate output folder
for each configuration. The defaults reproduce the scale/schedule of EchoScene's
`full_mp.yaml`; they are starting values, not calibrated DiffuScene weights.

Each constraint may override `start_ratio` and `interval`. With 1000 DDPM steps,
the defaults (`start_ratio: 0.95`, `interval: 5`) evaluate at t=49,44,...,4.
Global `strength` multiplies all losses; `grad_clip` limits each object's gradient
before the variance multiplier. Large weights can hit the clipping ceiling, so
increasing them need not increase the update proportionally.

## Losses and representation

- Collision: exact yaw-oriented bounding-box 3D IoU plus EchoScene's rotated-AABB
  penetration proxy. Thresholded penalties are averaged over all valid unordered
  pairs per scene. The penetration proxy can penalize nonintersecting OBBs.
- Room boundary: summed L1 excess of each rotated footprint outside the floor's
  X/Z bounding rectangle. The generation script centers the floor using
  `floor_plan_centroid`, matching DiffuScene translations. This rectangle does
  not represent concave walls, holes or doorways.
- Walkability: `type: center_penalty` uses
  `sum(exp(-distance_to_room_midpoint_squared / sigma))`, matching the currently
  enabled EchoScene walkable component while excluding empty slots.
  `type: edge_gaussian` instead combines a 64x64 floor-rectangle Gaussian heatmap
  with soft pairwise furniture repulsion. Its height cutoff uses center Y.
  These are walkability surrogates: neither verifies path connectivity. The
  OpenCV/A* pathfinding variant is not included.

Each loss is averaged across scenes, keeping batch-size scaling stable. Geometry
decoding uses DiffuScene's own JSON training statistics, converts half sizes to
full sizes, handles `[cos, sin]` or a normalized scalar angle, and follows the
mesh renderer's yaw convention. Empty slots are masked using the signed empty
class channel (`< 0` is furniture); configurations with a separate objectness
channel use its existing training convention (`>= 0` is furniture). Masks are
detached. Gradients flow only to translations and yaw. Sizes, classes,
objectness and object features are held constant in the physical losses.

## Sampling integration and logs

The denoiser runs without a gradient graph. At scheduled DDPM steps the predicted
clean layout is detached, made a gradient leaf, and differentiated through the
physical energies. The update is:

```text
guided_mean = model_mean - strength * model_variance * clipped_gradient_x0
```

The next DDPM sample uses this mean and the original noise variance. This applies
to eps, x0 and v model prediction types and to standard/progressive generation and
completion. Known completion boxes are restored by the existing sampler.
DDIM and arrangement with guidance raise explicit errors rather than silently
returning unguided output. Under `fixedsmall`, variance is zero at t=0.

Standard/progressive generation and completion also maintain an independent
unguided baseline path with the same initial noise and per-step noise. After
every step, sizes and all class/objectness/object-feature channels are restored
from that baseline. Losses use its clean sizes and semantic channels as well.
This prevents later denoiser steps from indirectly changing furniture count,
class, dimensions or shape features in response to the guided pose. Only
translations and yaw may differ. Guidance requires two denoiser evaluations per
step; no additional random draws are used (the inference denoiser must be in
evaluation mode and deterministic for exact comparison).

The reference is the baseline generated in the same run, not an existing JSON
or PLY. Matching an older baseline still requires the same checkpoint, data,
device, software environment, seed and job boundaries. Use a new output folder
when regenerating: existing per-job JSON is appended to and meshes are overwritten.
Frozen shape features select the same 3D-FUTURE model with an unchanged retrieval
database; existing PLY files are not read or reused.

`scripts/generate_diffusion.py` writes `guidance_losses_<start_index>.jsonl` into
the output directory. Each scene record contains its config, bounds, seed, and
per-step raw losses, weighted total and maximum gradient norm before clipping.
Absent loss keys mean that constraint was disabled or inactive at that step.
Zero loss/all-empty scenes remain safe for autograd. No mesh postprocessing is
performed by this guidance implementation.

For API use, pass `room_bounds=[xmin,zmin,xmax,zmax]` (or a `[B,4]` tensor) to
`generate_layout`, `generate_layout_progressive` or `complete_scene`. A fixed
`room_bounds` in the guidance YAML is a fallback for custom callers; missing
bounds raise an error when room/walkable guidance activates. Collision alone
does not require room bounds. Inspect step logs with
`network.diffusion.get_guidance_stats()` after sampling.

## Checks

Run the focused CPU suite without model weights or rendering dependencies:

```text
python -m unittest discover -s tests -p test_physical_guidance.py -v
```
