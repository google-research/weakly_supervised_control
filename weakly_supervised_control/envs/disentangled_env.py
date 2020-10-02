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
from typing import Dict, Optional, Tuple
import copy
import warnings

import cv2
from gym.spaces import Box
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.vae_wrapper import VAEWrappedEnv


class DisentangledEnv(VAEWrappedEnv):
    """This class wraps an image-based environment with a VAE and disentangled representation.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """

    def __init__(
            self,
            wrapped_env,
            vae,
            disentanglement_model=None,
            disentanglement_space: Tuple[Tuple[int], Tuple[int]] = None,
            disentanglement_indices: Tuple[int] = None,
            desired_goal_key: Optional[str] = None,
            *args,
            **kwargs,
    ):
        super().__init__(wrapped_env, vae, *args, **kwargs)

        self.disentanglement_model = disentanglement_model
        self.disentanglement_indices = disentanglement_indices
        self.desired_goal_key = desired_goal_key

        if disentanglement_space is not None:
            self.z_min = disentanglement_space.low
            self.z_max = disentanglement_space.high

            normalized_z_space = Box(
                0, 1, shape=disentanglement_space.shape, dtype=np.float32)
            self.observation_space.spaces['disentangled_desired_goal'] = normalized_z_space
            self.observation_space.spaces['disentangled_achieved_goal'] = normalized_z_space

    def try_render(self, obs, img_scale: float = 3):
        def imshow(img: np.ndarray, name: str):
            if img_scale is not None:
                width = int(img.shape[1] * img_scale)
                height = int(img.shape[0] * img_scale)
                img = cv2.resize(img, (width, height))
            cv2.imshow(name, img)
            cv2.waitKey(1)

        if self.render_rollouts:
            img = self.obs_to_image(obs['image_observation'])
            imshow(img, 'env')

            reconstruction = self._reconstruct_img(obs['image_observation']).transpose()
            imshow(reconstruction, 'env_reconstruction')

            init_img = self.obs_to_image(self._initial_obs['image_observation'])
            imshow(init_img, 'initial_state')

            init_reconstruction = self._reconstruct_img(
                    self._initial_obs['image_observation']
                ).transpose()
            imshow(init_reconstruction, 'init_reconstruction')

        if self.render_goals:
            goal = self.obs_to_image(obs['image_desired_goal'])
            imshow(goal, 'goal')

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs)
        reward = self.compute_reward(action, new_obs)
        self.try_render(new_obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        latent_obs = self._encode_one(obs[self.vae_input_observation_key])
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = latent_obs
        obs['observation'] = latent_obs
        obs['achieved_goal'] = latent_obs

        if self.disentanglement_model is not None:
            obs['disentangled_achieved_goal'] = self._disentangle_obs(
                obs[self.vae_input_observation_key])[0]
        obs = {**obs, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        if 'latent_desired_goal' in self.desired_goal:
            latent_distribution_params = self.vae.encode(
                ptu.from_numpy(obs[self.vae_input_observation_key].reshape(1, -1)))
            latent_obs, logvar = ptu.get_numpy(
                latent_distribution_params[0])[0], ptu.get_numpy(
                    latent_distribution_params[1])[0]
            # assert (latent_obs == obs['latent_observation']).all()
            latent_goal = self.desired_goal['latent_desired_goal']
            var = np.exp(logvar.flatten())
            var = np.maximum(var, self.reward_min_variance)
            dist = latent_goal - latent_obs
            err = dist * dist / 2 / var
            mdist = np.sum(err)  # mahalanobis distance
            info["vae_mdist"] = mdist
            info["vae_success"] = 1 if mdist < self.epsilon else 0
            info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
            info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
            info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    """
    Multitask functions
    """

    def sample_goals(self, batch_size):
        if self._goal_sampling_mode == 'custom_goal_sampler':
            sampled_goals = self.custom_goal_sampler(batch_size)
            # ensures goals are encoded using latest vae
            if 'image_desired_goal' in sampled_goals:
                sampled_goals['latent_desired_goal'] = self._encode(
                    sampled_goals['image_desired_goal'])

                if self.disentanglement_model is not None:
                    sampled_goals['disentangled_desired_goal'] = self._disentangle_obs(
                        sampled_goals['image_desired_goal'])
            return sampled_goals
        elif self._goal_sampling_mode == 'presampled':
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx]
                for k, v in self._presampled_goals.items()
            }
            # ensures goals are encoded using latest vae
            if 'image_desired_goal' in sampled_goals:
                sampled_goals['latent_desired_goal'] = self._encode(
                    sampled_goals['image_desired_goal'])
                if self.disentanglement_model is not None:
                    sampled_goals['disentangled_desired_goal'] = self._disentangle_obs(
                        sampled_goals['image_desired_goal'])
            return sampled_goals
        elif self._goal_sampling_mode == 'env':
            goals = self.wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(goals[self.vae_input_desired_goal_key])
        elif self._goal_sampling_mode == 'reset_of_env':
            assert batch_size == 1
            goal = self.wrapped_env.get_goal()
            goals = {k: v[None] for k, v in goal.items()}
            latent_goals = self._encode(goals[self.vae_input_desired_goal_key])
        elif self._goal_sampling_mode == 'vae_prior':
            goals = {}
            latent_goals = self._sample_vae_prior(batch_size)
        elif self._goal_sampling_mode == 'observation_space':
            goals = []
            for _ in range(batch_size):
                goal = self.observation_space.spaces[self.desired_goal_key].sample(
                )
                if len(goal.shape) == 0:
                    goal = np.expand_dims(goal, 0)
                goals.append(goal)
            goals = np.stack(goals)
            return {'desired_goal': goals, self.desired_goal_key: goals}
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))

        if self._decode_goals:
            decoded_goals = self._decode(latent_goals)
        else:
            decoded_goals = None
        image_goals, proprio_goals = self._image_and_proprio_from_decoded(
            decoded_goals)

        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        if proprio_goals is not None:
            goals['proprio_desired_goal'] = proprio_goals
        if image_goals is not None:
            goals['image_desired_goal'] = image_goals

            if self.disentanglement_model is not None:
                goals['disentangled_desired_goal'] = self._disentangle_obs(
                    image_goals)
        if decoded_goals is not None:
            goals[self.vae_input_desired_goal_key] = decoded_goals
        return goals

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {k: v[None] for k, v in obs.items()}
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        if self.reward_type == 'latent_distance':
            dist = np.linalg.norm(obs['latent_desired_goal'] - obs['latent_achieved_goal'],
                                  ord=self.norm_order,
                                  axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            return -np.abs(obs['latent_desired_goal'] - obs['latent_achieved_goal'])
        elif self.reward_type == 'latent_sparse':
            dist = np.linalg.norm(obs['latent_desired_goal'] - obs['latent_achieved_goal'],
                                  ord=self.norm_order,
                                  axis=1)
            reward = 0 if dist < self.epsilon else -1
            return reward
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            if self.disentanglement_indices is not None:
                achieved_goals = achieved_goals[:,
                                                self.disentanglement_indices]
                desired_goals = desired_goals[:, self.disentanglement_indices]

            return -np.linalg.norm(
                desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'disentangled_distance':
            return -np.linalg.norm(
                obs['disentangled_desired_goal'] -
                obs['disentangled_achieved_goal'],
                ord=self.norm_order, axis=1)
        elif self.reward_type == 'latent_and_disentangled_distance':
            latent_dist = np.linalg.norm(
                obs['latent_desired_goal'] - obs['latent_achieved_goal'],
                ord=self.norm_order,
                axis=1)
            disentangled_dist = np.linalg.norm(
                obs['disentangled_desired_goal'] -
                obs['disentangled_achieved_goal'],
                ord=self.norm_order, axis=1)
            return -(latent_dist + disentangled_dist)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["vae_mdist", "vae_success", "vae_dist"]:
            if stat_name_in_paths in paths[0]['env_infos'][0]:
                stats = get_stat_in_paths(
                    paths, 'env_infos', stat_name_in_paths)
                statistics.update(
                    create_stats_ordered_dict(
                        stat_name_in_paths,
                        stats,
                        always_show_all_stats=True,
                    ))
                final_stats = [s[-1] for s in stats]
                statistics.update(
                    create_stats_ordered_dict(
                        "Final " + stat_name_in_paths,
                        final_stats,
                        always_show_all_stats=True,
                    ))
        return statistics

    """
    Other functions
    """

    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'custom_goal_sampler', 'presampled', 'vae_prior', 'env',
            'reset_of_env', 'observation_space',
        ], "Invalid env mode"
        self._goal_sampling_mode = mode
        if mode == 'custom_goal_sampler':
            test_goals = self.custom_goal_sampler(1)
            if test_goals is None:
                self._goal_sampling_mode = 'vae_prior'
                warnings.warn(
                    "self.goal_sampler returned None. " +
                    "Defaulting to vae_prior goal sampling mode"
                )

    def _disentangle_obs(self, imgs):
        imgs = imgs.reshape((-1, self.channels, self.imsize, self.imsize)
                            ).transpose((0, 3, 2, 1))
        zs = self.disentanglement_model.enc_eval(imgs).numpy()
        if self.disentanglement_indices is not None:
            zs = zs[:, self.disentanglement_indices]
            if len(zs.shape) == 1:
                zs = np.expand_dims(zs, 0)

        zs = (zs - self.z_min) / (self.z_max - self.z_min)
        return zs

    def __getstate__(self):
        state = super().__getstate__()
        state = copy.copy(state)
        state['disentanglement_model'] = None
        state['_custom_goal_sampler'] = None
        warnings.warn(
            'VAEWrapperEnv.custom_goal_sampler and disentanglement_model are not saved.')
        return state

    def __setstate__(self, state):
        warnings.warn(
            'VAEWrapperEnv.custom_goal_sampler and disentanglement_model were not loaded.')
        super().__setstate__(state)
