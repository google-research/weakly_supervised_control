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
from typing import Dict, List
import numpy as np

from multiworld.core.image_env import normalize_image
from rlkit.data_management.obs_dict_replay_buffer import flatten_dict
from rlkit.data_management.online_vae_replay_buffer import OnlineVaeRelabelingBuffer
from weakly_supervised_control.envs.disentangled_env import DisentangledEnv
from rlkit.torch.vae.vae_trainer import relative_probs_from_log_probs


class ReplayBuffer(OnlineVaeRelabelingBuffer):

    def __init__(
            self,
            vae,
            *args,
            encoded_obs_key='latent_observation',
            internal_keys: List[str] = [],
            snapshot: Dict = None,
            **kwargs
    ):
        if encoded_obs_key not in internal_keys:
            internal_keys.append(encoded_obs_key)
        super().__init__(vae, internal_keys=internal_keys, *args, **kwargs)
        assert isinstance(self.env, DisentangledEnv)
        self.encoded_obs_key = encoded_obs_key

        if snapshot is not None:
            self._obs = snapshot['_obs']
            self._actions = snapshot['_actions']
            self._next_obs = snapshot['_next_obs']
            self._terminals = snapshot['_terminals']
            self._top = snapshot['_top']
            self._size = snapshot['_size']
            self._idx_to_future_obs_idx = snapshot['_idx_to_future_obs_idx']

    def add_decoded_vae_goals_to_path(self, path):
        # decoding the self-sampled vae images should be done in batch (here)
        # rather than in the env for efficiency
        if 'latent_desired_goal' in path['observations'][0].keys():
            desired_encoded_goals = flatten_dict(
                path['observations'],
                ['latent_desired_goals']
            )['latent_desired_goals']
            desired_decoded_goals = self.env._decode(desired_encoded_goals)
            desired_decoded_goals = desired_decoded_goals.reshape(
                len(desired_decoded_goals),
                -1
            )
            for idx, next_obs in enumerate(path['observations']):
                path['observations'][idx][self.decoded_desired_goal_key] = \
                    desired_decoded_goals[idx]
                path['next_observations'][idx][self.decoded_desired_goal_key] = \
                    desired_decoded_goals[idx]

    def add_path(self, path):
        self.add_decoded_vae_goals_to_path(path)
        super().add_path(path)

    def refresh_latents(self, epoch):
        self.epoch = epoch
        self.skew = (self.epoch > self.start_skew_epoch)
        batch_size = 512
        next_idx = min(batch_size, self._size)

        if self.exploration_rewards_type == 'hash_count':
            # you have to count everything then compute exploration rewards
            cur_idx = 0
            next_idx = min(batch_size, self._size)
            while cur_idx < self._size:
                idxs = np.arange(cur_idx, next_idx)
                normalized_imgs = (
                    normalize_image(self._next_obs[self.decoded_obs_key][idxs])
                )
                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, self._size)

        cur_idx = 0
        obs_sum = np.zeros(self.vae.representation_size)
        obs_square_sum = np.zeros(self.vae.representation_size)
        while cur_idx < self._size:
            idxs = np.arange(cur_idx, next_idx)
            self._obs[self.encoded_obs_key][idxs] = \
                self.env._encode(
                    normalize_image(self._obs[self.decoded_obs_key][idxs])
            )
            self._next_obs[self.encoded_obs_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.decoded_obs_key][idxs])
            )
            # WARNING: we only refresh the desired/achieved latents for
            # "next_obs". This means that obs[desired/achieve] will be invalid,
            # so make sure there's no code that references this.
            # TODO: enforce this with code and not a comment
            if 'latent_desired_goal' in self._next_obs:
                self._next_obs['latent_desired_goal'][idxs] = \
                    self.env._encode(
                        normalize_image(
                            self._next_obs[self.decoded_desired_goal_key][idxs])
                )
            if 'latent_achieved_goal' in self._next_obs:
                self._next_obs['latent_achieved_goal'][idxs] = \
                    self.env._encode(
                        normalize_image(
                            self._next_obs[self.decoded_achieved_goal_key][idxs])
                )
            normalized_imgs = (
                normalize_image(self._next_obs[self.decoded_obs_key][idxs])
            )
            if self._give_explr_reward_bonus:
                rewards = self.exploration_reward_func(
                    normalized_imgs,
                    idxs,
                    **self.priority_function_kwargs
                )
                self._exploration_rewards[idxs] = rewards.reshape(-1, 1)
            if self._prioritize_vae_samples:
                if (
                        self.exploration_rewards_type == self.vae_priority_type
                        and self._give_explr_reward_bonus
                ):
                    self._vae_sample_priorities[idxs] = (
                        self._exploration_rewards[idxs]
                    )
                else:
                    self._vae_sample_priorities[idxs] = (
                        self.vae_prioritization_func(
                            normalized_imgs,
                            idxs,
                            **self.priority_function_kwargs
                        ).reshape(-1, 1)
                    )
            obs_sum += self._obs[self.encoded_obs_key][idxs].sum(axis=0)
            obs_square_sum += np.power(
                self._obs[self.encoded_obs_key][idxs], 2).sum(axis=0)

            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, self._size)
        self.vae.dist_mu = obs_sum/self._size
        self.vae.dist_std = np.sqrt(
            obs_square_sum/self._size - np.power(self.vae.dist_mu, 2))

        if self._prioritize_vae_samples:
            """
            priority^power is calculated in the priority function
            for image_bernoulli_prob or image_gaussian_inv_prob and
            directly here if not.
            """
            if self.vae_priority_type == 'vae_prob':
                self._vae_sample_priorities[:self._size] = relative_probs_from_log_probs(
                    self._vae_sample_priorities[:self._size]
                )
                self._vae_sample_probs = self._vae_sample_priorities[:self._size]
            else:
                self._vae_sample_probs = self._vae_sample_priorities[:self._size] ** self.power
            p_sum = np.sum(self._vae_sample_probs)
            assert p_sum > 0, "Unnormalized p sum is {}".format(p_sum)
            self._vae_sample_probs /= np.sum(self._vae_sample_probs)
            self._vae_sample_probs = self._vae_sample_probs.flatten()
