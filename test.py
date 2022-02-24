# import blenderproc as bproc
import json
import numpy as np

from tqdm import tqdm

# with open('output/train_pbr/000000/scene_gt.json') as f:
#     a = json.load(f)
#     print(a['0'])
with open('output/train_pbr/000001/scene_gt.json') as f:
    b = json.load(f)
    print(b['0'])

# a = np.load('output/pose/pose0.npy')
# print(a)
b = np.load('output/pose/pose1000.npy')
print(b)
