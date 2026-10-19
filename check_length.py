import os
import sys
sys.path.append('scripts')
from training_utils import load_config
from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded

config = load_config('config/uncond/diffusion_livingrooms_instancond_lat32_v.yaml')
config['data']['encoding_type'] = config['data']['encoding_type'].replace('text', 'textfix')
config['data']['encoding_type'] = config['data']['encoding_type'] + '_no_prm'
raw_dataset, dataset = get_dataset_raw_and_encoded(
    config['data'],
    filter_fn=filter_function(
        config['data'],
        split=['test']
    ),
    split=['test']
)
print('NUMBER_OF_TEST_SCENES:', len(dataset))
