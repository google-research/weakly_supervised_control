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

"""Environment wrappers for augmenting environment data."""

import abc
import collections.abc
from typing import Any, Callable, Dict, List, Tuple, Type

import gym
from gym import spaces
import numpy as np


def create_wrapped_env(env_cls: Type, env_kwargs: Dict[str, Any],
                       wrappers: List[Tuple[Type, Dict[str, Any]]]):
    """Creates a wrapped environment."""
    env = env_cls(**env_kwargs)
    for wrapper_cls, wrapper_kwargs in wrappers:
        env = wrapper_cls(env, **wrapper_kwargs)
    return env


class GoalFactorWrapper(gym.Wrapper, metaclass=abc.ABCMeta):
    """Wraps a goal-based environment."""

    STATE_OBS_KEYS = ('state_desired_goal', 'state_achieved_goal',
                      'state_observation')

    GOAL_KEYS = ('state_desired_goal', 'desired_goal')

    def __init__(self, env: gym.Env, factor_space: spaces.Box):
        super().__init__(env)
        self.factor_space = factor_space
        self.factor_size = len(np.atleast_1d(factor_space.low))
        self.current_factor_value = None

        # Augment the observation space.
        if isinstance(self.observation_space, spaces.Dict):
            for key in self.STATE_OBS_KEYS:
                prev_goal_space = self.observation_space.spaces[key]
                self.observation_space.spaces[key] = spaces.Box(
                    low=np.append(prev_goal_space.low, factor_space.low),
                    high=np.append(prev_goal_space.high, factor_space.high),
                    dtype=prev_goal_space.dtype,
                )
        else:
            print('WARNING: {} observation_space is not Dict: {}'.format(
                self.__class__.__name__, self.observation_space))

    @abc.abstractmethod
    def _set_to_factor(self, value: np.ndarray):
        """Sets to the given factor."""

    def randomize_factor(self):
        factor = self._sample_factors(1)[0]
        self._set_and_cache_factor(factor)

    def reset(self):
        obs = self.env.reset()
        self.randomize_factor()
        obs = self._add_factor_to_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._add_factor_to_obs(obs)
        return obs, reward, done, info

    def set_to_goal(self, goal):
        goal = goal.copy()
        # Extract out the factor before calling the sub-env's set_to_goal.
        for goal_key in self.GOAL_KEYS:
            if goal_key in goal:
                factor = goal[goal_key][-self.factor_size:]
                goal[goal_key] = (goal[goal_key][:-self.factor_size])
        self.env.set_to_goal(goal)
        self._set_and_cache_factor(factor)

    def set_goal(self, goal):
        goal = goal.copy()
        # Extract out the factor before calling the sub-env's set_to_goal.
        for goal_key in self.GOAL_KEYS:
            if goal_key in goal:
                factor = goal[goal_key][-self.factor_size:]
                goal[goal_key] = (goal[goal_key][:-self.factor_size])
        self.env.set_goal(goal)
        self._set_and_cache_factor(factor)

    def sample_goals(self, batch_size):
        goals = self.env.sample_goals(batch_size)
        # Add the factor value.
        factor = self._sample_factors(batch_size)
        for goal_key in self.GOAL_KEYS:
            desired_goal = np.concatenate([goals[goal_key], factor], axis=1)
            goals[goal_key] = desired_goal
        return goals

    def compute_rewards(self, actions, obs):
        obs = self._remove_factor_from_obs(obs)
        return self.env.compute_rewards(actions, obs)

    def _sample_factors(self, batch_size: int) -> np.ndarray:
        """Returns factor values."""
        return np.stack([
            np.atleast_1d(self.factor_space.sample())
            for i in range(batch_size)
        ])

    def _set_and_cache_factor(self, value: np.ndarray):
        if self.factor_space.low.shape == ():
            value = np.array(value.item())
        # if not self.factor_space.contains(value):
        #     print('WARNING: {} does not contain {}'.format(
        #         self.factor_space, value))
        self._set_to_factor(value)
        self.current_factor_value = value

    def _add_factor_to_obs(self, obs):
        assert self.current_factor_value is not None
        if not isinstance(obs, collections.abc.Mapping):
            return obs
        obs = obs.copy()
        for key in self.STATE_OBS_KEYS:
            if key not in obs:
                print('WARNING: {} not in obs'.format(key))
                continue
            obs[key] = np.append(obs[key], self.current_factor_value)
        return obs

    def _remove_factor_from_obs(self, obs):
        obs = obs.copy()
        if not isinstance(obs, collections.abc.Mapping):
            return obs
        for key in self.STATE_OBS_KEYS:
            if key not in obs:
                print('WARNING: {} not in obs'.format(key))
                continue
            obs_value = obs[key]
            assert obs_value.ndim in (1, 2), obs_value.ndim
            if obs_value.ndim == 1:
                obs[key] = obs_value[:-self.factor_size]
            else:
                obs[key] = obs_value[:, :-self.factor_size]
        return obs

    def __getattr__(self, name: str):
        return getattr(self.env, name)


class MujocoRandomLightsWrapper(GoalFactorWrapper):
    """Wrapper over MuJoCo environments that modifies lighting."""

    def __init__(self,
                 env: gym.Env,
                 diffuse_range: Tuple[float, float] = (0.2, 0.8)):
        """Creates a new wrapper."""
        super().__init__(
            env,
            factor_space=spaces.Box(low=np.array(diffuse_range[0]),
                                    high=np.array(diffuse_range[1]),
                                    dtype=np.float32),
        )
        self.model = self.unwrapped.model
    
    @property
    def factor_names(self):
        factor_names = self.unwrapped.factor_names
        return factor_names + ['light']

    def _set_to_factor(self, value: float):
        """Sets to the given factor."""
        self.model.vis.headlight.ambient[:] = np.full((3, ), value)
        self.model.vis.headlight.diffuse[:] = np.full((3, ), value)

    def __getattr__(self, name: str):
        return getattr(self.env, name)


class MujocoRandomColorWrapper(GoalFactorWrapper):
    """Wrapper over MuJoCo environments that modifies lighting."""

    def __init__(self,
                 env: gym.Env,
                 color_choices: List[Tuple[float, float, float, float]],
                 geom_names: List[str] = None,
                 site_names: List[str] = None):
        """Creates a new wrapper."""
        super().__init__(
            env,
            factor_space=spaces.Box(
                low=np.array(0, dtype=int),
                high=np.array(len(color_choices) - 1, dtype=int),
                dtype=int,
            ),
        )
        self.model = self.unwrapped.model
        self.geom_ids = [
            self.model.geom_name2id(name) for name in geom_names or []
        ]
        self.site_ids = [
            self.model.site_name2id(name) for name in site_names or []
        ]
        self.color_choices = color_choices

    @property
    def factor_names(self):
        factor_names = self.unwrapped.factor_names
        return factor_names + ['table_color', 'obj_color']

    def _set_to_factor(self, value: int):
        """Sets to the given factor."""
        color = self.color_choices[int(value)]
        for geom_id in self.geom_ids:
            self.model.geom_rgba[geom_id, :] = color
        for site_id in self.site_ids:
            self.model.site_rgba[site_id, :] = color

    def __getattr__(self, name: str):
        return getattr(self.env, name)


if __name__ == '__main__':
    from weakly_supervised_control.envs import register_all_envs
    import gym
    register_all_envs()
    env = gym.make('SawyerPickupRandomLightsColorsEnv-v1')
    # env = MujocoRandomColorWrapper(
    #     env,
    #     geom_names=['tableTop'],
    #     color_choices=[
    #         (.6, .6, .5, 1),
    #         (1., .6, .5, 1),
    #         (.6, 1., .5, 1),
    #         (.6, 1., 1., 1),
    #         (1., 1., .5, 1),
    #     ],
    # )

    for e in range(10):
        obs = env.reset()
        # assert len(obs['state_desired_goal']) == 5, obs['state_desired_goal']

        for _ in range(100):
            env.step(env.action_space.sample())
            # env.randomize_factor()
            env.render()
