from simple_3dviz import Scene, Mesh
from pyrr import Matrix44
import numpy as np

scene = Scene(size=(256, 256), background=[1, 1, 1, 1])
scene.up_vector = (0, 0, -1)
scene.camera_target = (0, 0, 0)
scene.camera_position = (0, 4, 0)
scene.light = (0, 4, 0)
scene.camera_matrix = Matrix44.orthogonal_projection(left=-3.1, right=3.1, bottom=3.1, top=-3.1, near=0.1, far=6)

mesh = Mesh.from_file('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013.ply')
mesh.mode = "shading"
scene.add(mesh)
scene.render()

from simple_3dviz.utils import save_frame
save_frame('test_ply.png', scene.frame)
