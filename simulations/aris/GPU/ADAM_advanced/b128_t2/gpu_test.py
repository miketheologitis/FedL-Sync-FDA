# coding: utf-8
# In[1]:

import os

slurm_localid = int(os.environ.get('SLURM_LOCALID'))
    
os.environ['CUDA_VISIBLE_DEVICES'] = str(slurm_localid)

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


print("GPU Setup:")
print(tf.config.list_physical_devices('GPU'))
print()



import json
import os

slurm_procid = os.environ.get('SLURM_PROCID')


with open('combinations.json', 'r') as f:
    all_combinations = json.load(f)


print(f"Slurm procid: {slurm_procid}   type(slurm_procid) : {type(slurm_procid)}  slurm_localid : {slurm_localid}")

print(all_combinations[int(slurm_procid)])