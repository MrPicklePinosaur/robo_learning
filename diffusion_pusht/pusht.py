from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import collections

import gymnasium as gym
import gym_pusht

from dataloader import PushTImageDataset

DATASET_PATH='data/pusht_cchi_v7_replay.zarr'
BATCHS_SIZE=64

# Data loader =====

# some parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset = PushTImageDataset(
    dataset_path=DATASET_PATH,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCHS_SIZE,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

# Visual encoder
# TODO skip this for now

# Setup gym environment
env = gym.make("gym_pusht/PushT-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
