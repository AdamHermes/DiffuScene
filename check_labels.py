import json
import numpy as np
d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')
labels = np.argmax(d1['class_labels'][idx], axis=1)
print(labels)
