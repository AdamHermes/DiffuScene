import json, numpy as np; d=json.load(open('bedroom/collision_params_merged.json')); 
for i in range(5): print(np.mean(d['translations'][i], axis=0))
