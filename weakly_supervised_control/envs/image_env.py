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
from typing import Dict, List, Tuple
from collections import OrderedDict
from numbers import Number
import random
import warnings

import cv2
import gym
#from gym.spaces import Box, Dict
import numpy as np
from PIL import Image

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces
# from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

from weakly_supervised_control.envs.env_util import normalize_image, unormalize_image, concat_images

def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]


class ImageEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env: gym.Env,
            x_shape: Tuple[int],
            normalize_image_observations: bool = True,
            presampled_goals: Dict[str, np.ndarray] = None,
            reward_type: str = 'wrapped_env',
            threshold: float = 10,
            recompute_reward: bool = True,
    ):
        """
        :param wrapped_env:
        :param imsize:
        :param reward_type:
        :param threshold:
        :param image_length:
        :param presampled_goals:
        """
        self.quick_init(locals())
        super().__init__(wrapped_env)

        # Get observation shape.
        assert len(x_shape) == 3, x_shape
        assert x_shape[0] == x_shape[1]
        self.imsize = x_shape[0]
        self.channels = x_shape[-1]

        self.normalize_image_observations = normalize_image_observations
        self.wrapped_env.hide_goal_markers = True
        self.recompute_reward = recompute_reward

        # This is torch format rather than PIL image
        self.image_shape = (self.imsize, self.imsize)
        self.image_length = self.channels * self.imsize * self.imsize
        img_space = gym.spaces.Box(
            0, 1, (self.image_length,), dtype=np.float32)
        self._image_goal = img_space.sample()  # has to be done for presampling
        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['observation'] = img_space
        spaces['desired_goal'] = img_space
        spaces['achieved_goal'] = img_space
        spaces['image_observation'] = img_space
        spaces['image_desired_goal'] = img_space
        spaces['image_achieved_goal'] = img_space
        self.observation_space = gym.spaces.Dict(spaces)

        self.action_space = self.wrapped_env.action_space
        self.reward_type = reward_type
        self.threshold = threshold
        self._last_image = None

        self._presampled_goals = None
        if presampled_goals is not None:
            assert 'image_desired_goal' in presampled_goals, presampled_goals.keys()

            # Normalize goal images.
            image_goals = presampled_goals['image_desired_goal'].copy()
            image_goals_max = np.max(image_goals)
            if image_goals_max > 1:
                assert image_goals_max <= 255, image_goals_max
                assert np.min(image_goals) >= 0, np.min(image_goals)
                image_goals = normalize_image(image_goals)

            if len(image_goals[0].shape) != 1:
                # Flatten images.
                presampled_goals['image_desired_goal'] = np.array(
                    [self._flatten_image(image) for image in image_goals])
            self._presampled_goals = presampled_goals
            self.num_goals_presampled = presampled_goals['image_desired_goal'].shape[0]

    def _flatten_image(self, image: np.ndarray) -> np.ndarray:
        # Flatten the image from (w, h, 3n) to (3n * w * h).
        image = image.transpose()
        assert image.shape[0] == self.channels, image.shape
        return image.flatten()
    
    def  _unflatten_image(self, image: np.ndarray) -> np.ndarray:
        image = image.reshape(
                self.channels,
                self.imsize,
                self.imsize,
            ).transpose()
        assert image.shape[2] == self.channels,  image.shape
        return image

    def obs_to_image(self, image_obs: np.ndarray) -> np.ndarray:
        # Input obs is of shape (3n * w * h)
        image_obs = self._unflatten_image(image_obs)  # (w, h, 3n)
        images = np.split(image_obs, self.channels / 3, axis=2)  # n images of shape (w, h, 3)
        images = np.concatenate([np.expand_dims(img, 0) for img in images])  # (n, w, h, 3)
        stacked_images = concat_images(images)  # (n * w, h, 3)
        return stacked_images

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        if self.recompute_reward:
            reward = self.compute_reward(action, new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def _update_info(self, info, obs):
        achieved_goal = obs['image_achieved_goal']
        desired_goal = self._image_goal
        image_dist = np.linalg.norm(achieved_goal - desired_goal)
        image_success = (image_dist < self.threshold).astype(float) - 1
        info['image_dist'] = image_dist
        info['image_success'] = image_success

    def reset(self):
        obs = self.wrapped_env.reset()
        goal = self.sample_goal()
        self.set_goal(goal)
        for key in goal:
            obs[key] = goal[key]
        return self._update_obs(obs)

    def _get_obs(self):
        return self._update_obs(self.wrapped_env._get_obs())

    def _update_obs(self, obs):
        image_obs = self._wrapped_env.get_image(
            width=self.imsize,
            height=self.imsize,
        )
        self._last_image = image_obs
        if self.normalize_image_observations:
            image_obs = normalize_image(image_obs)
        image_obs = self._flatten_image(image_obs)
        obs['image_observation'] = image_obs
        obs['image_desired_goal'] = self._image_goal
        obs['image_achieved_goal'] = image_obs
        obs['observation'] = image_obs
        obs['desired_goal'] = self._image_goal
        obs['achieved_goal'] = image_obs
        return obs

    def render(self, mode='wrapped'):
        if mode == 'wrapped':
            self.wrapped_env.render()
        elif mode == 'cv2':
            if self._last_image is None:
                self._last_image = self._wrapped_env.get_image(
                    width=self.imsize,
                    height=self.imsize,
                )
            cv2.imshow('ImageEnv', self._last_image)
            cv2.waitKey(1)
        else:
            raise ValueError("Invalid render mode: {}".format(mode))

    """
    Multitask functions
    """

    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._image_goal
        goal['image_desired_goal'] = self._image_goal
        return goal

    def set_goal(self, goal):
        ''' Assume goal contains both image_desired_goal and any goals required for wrapped envs'''
        self._image_goal = goal['image_desired_goal']
        self.wrapped_env.set_goal(goal)

    def sample_goals(self, batch_size):
        assert self._presampled_goals is not None
        idx = np.random.randint(0, self.num_goals_presampled, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals

    def compute_rewards(self, actions, obs):
        if self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)

        dist = np.linalg.norm(obs['achieved_goal'] -
                              obs['desired_goal'], axis=1)
        if self.reward_type == 'image_distance':
            return -dist
        elif self.reward_type == 'image_sparse':
            return -(dist > self.threshold).astype(float)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, **kwargs):
        if len(paths) == 0:
            return {}

        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["image_dist", "image_success"]:
            stats = get_stat_in_paths(
                paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics
