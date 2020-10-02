# Copyright 2020 The Weakly-Supervised Control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ground-truth data implementation for RL environments."""

from collections import defaultdict
from itertools import product, combinations
from typing import Any, Dict, Iterable, List, Optional

import gym
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile


class NpGroundTruthData():
    @classmethod
    def load(cls, path: str, image_size: int = 48, **kwargs):
        with gfile.GFile(path, 'rb') as fin:
            data = np.load(fin, allow_pickle=True)
        if hasattr(data, 'item'):
            data = data.item()

        images = data['image_desired_goal']
        if len(images.shape) == 2:  # For backward compatibility
            images = images.reshape((-1, 3, image_size, image_size))
            images = np.transpose(images, (0, 3, 2, 1))

        return cls(data=images, factors=data['state_desired_goal'],
                   factor_names=data.get('factor_names', None), **kwargs)

    def save(self, path: str):
        """Saves the data to a file path."""
        assert self.data.shape[-1] == 3
        #images = np.transpose(self.data, (0, 3, 2, 1))
        # images = images.reshape((images.shape[0], -1))
        np.savez_compressed(path,
                            state_desired_goal=self.factors,
                            image_desired_goal=self.data,
                            factor_names=self.factor_names)

    @classmethod
    def sample_env_goals(
        cls,
        env: gym.Env,
        num_samples: int = 1e3,
        image_size: int = 48,
    ):
        factor_names = None
        if hasattr(env, 'factor_names'):
            factor_names = env.factor_names

        print('Sampling image observations...')
        goals = env.sample_goals(num_samples)
        data = []
        for i in range(num_samples):
            goal = {k: v[i] for k, v in goals.items()}
            env.set_to_goal(goal)
            obs = env.get_image(image_size, image_size)
            obs = obs / 255.0
            data.append(obs)
        data = np.array(data)
        factors = goals['state_desired_goal']

        return cls(data, factors, factor_names=factor_names)

    def __init__(self,
                 data: np.ndarray,
                 factors: np.ndarray,
                 factor_names: List[str],
                 stack_images: bool = False,
                 ):
        if stack_images:
            # Data should be of shape (batch_size, num_images, w, h, 3)
            assert len(data.shape) == 5, data.shape
            assert data.shape[-1] == 3, data.shape
            data = np.array([np.concatenate(image, axis=2) for image in data])
        self.data = data
        self.factors = factors
        self.factor_names = None
        self._process_factors(factor_names)

    """GroundTruthData implementation using Numpy arrays. Only rank-pairing is supported."""

    def _process_factors(self,
                         factor_names: List[str] = None):
        if factor_names is not None:
            self.factor_names = factor_names
        else:
            self.factor_names = list(range(self.factors.shape[1]))
        assert len(self.factor_names) == self.factors.shape[1], len(
            self.factor_names)

        # Normalize factor values.
        self.factors = (
            (self.factors - np.min(self.factors)) / np.ptp(self.factors))

        # Get unique factor values.
        self.factor_values = []
        self.factor_sizes = []
        num_factors = self.factors.shape[1]
        for i in range(num_factors):
            unique_vals = np.unique(self.factors[:, i])
            self.factor_values.append(unique_vals)
            self.factor_sizes.append(len(unique_vals))

    @property
    def size(self) -> int:
        """Returns the size of the dataset."""
        return len(self.data)

    @property
    def num_factors(self) -> int:
        return self.factors.shape[1]

    @property
    def factors_num_values(self) -> List[int]:
        return self.factor_sizes

    @property
    def observation_shape(self):
        """Returns the observation shape of the dataset."""
        return self.data[0].shape

    def sample(self, batch_size: int, random_state: Optional[np.random.RandomState] = None):
        if batch_size > self.size:
            idx = range(self.size)
        else:
            idx = np.random.choice(range(self.size), batch_size, replace=False)
        return self.data[idx]

    def sample_rank_pair(self, batch_size: int, masks: np.ndarray = None, random_state: Optional[np.random.RandomState] = None):
        """
        Returns observations (x1, x2) and a boolean tensor (y) indicating rank.
        """
        if masks is None:
            masks = np.arange(self.num_factors)
        idx = np.random.choice(range(self.size), batch_size * 2, replace=False)
        idx1, idx2 = np.split(idx, 2)
        x1, x2 = self.data[idx1], self.data[idx2]
        factor1, factor2 = self.factors[idx1], self.factors[idx2]
        y = np.array(factor1 > factor2, dtype=np.float32)[:, masks]
        return x1, x2, y

    def sample_match_pair(self, batch_size: int, random_state: Optional[np.random.RandomState] = None):
        raise NotImplementedError()
