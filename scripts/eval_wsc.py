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
r"""Runs a trained policy.

python -m scripts.eval_wsc --input 
"""
from typing import Tuple
import click
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from tensorflow.io import gfile
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.rollout_functions import rollout

from weakly_supervised_control.experiment_utils import load_experiment, process_image_trajectory

GS_DIR = 'gs://brain-adept-gs/users/lslee/job/'


@click.command()
@click.option('--input', '-i', type=str)
@click.option('--output', '-o',
              type=str,
              default='/tmp/weakly_supervised_control/eval_wsc',
              help="Output directory")
@click.option('--max-path-length', default=50, type=int)
@click.option('--grid-size', default=5, type=int)
@click.option('--factor-indices', default=None, type=str, help="Comma-separated list of ints. Only used for VAE latent space visualization.")
@click.option('--render', default=False, is_flag=True)
@click.option('--cpu', default=False, is_flag=True)
def main(input: str, output: str,
         max_path_length: int,
         grid_size: int,
         factor_indices: str,
         render: bool, cpu: bool):
    policy, variant, env = load_experiment(input)

    # Get goal space.
    desired_goal_key = variant['her_variant']['desired_goal_key']
    exploration_goal_sampler = variant['her_variant']['exploration_goal_sampler']
    if exploration_goal_sampler == 'observation_space':
        # Case 1: Disentangled latent space
        print('Using disentangled space')
        assert desired_goal_key in [
            'disentangled_desired_goal', 'state_desired_goal']
        goal_space = env.observation_space.spaces[desired_goal_key]
        assert goal_space.shape == (2,)
        xy = np.linspace(goal_space.low, goal_space.high, grid_size)

        goals = []
        for y in xy[:, 1]:
            goals.append([np.array([x, y]) for x in xy[:, 0]])
        goals = np.array(goals)
    else:
        # Case 2: VAE latent space
        from scipy.stats import pearsonr
        from weak_disentangle.train_utils import load_dset
        print('Using VAE latent space')

        factor_indices = list(map(int, factor_indices.split(',')))
        assert len(factor_indices) == 2

        # Load dataset.
        dset = load_dset(variant['env']['train_dataset'])
        factors = dset.factors
        data = dset.data.transpose(0, 3, 2, 1)
        data = data.reshape((data.shape[0], -1))

        mu, _ = env.vae.encode(ptu.from_numpy(data))
        mu = mu.detach().numpy()
        z_max = np.max(mu, axis=0)
        z_min = np.min(mu, axis=0)

        # Find latent dimensions that has the highest correlation with factor_indices.
        best_latent_indices = []
        best_corr_coeffs = []
        for factor_idx in factor_indices:
            factor_vals = factors[:, factor_idx]

            best_latent_idx = None
            best_corr = 0
            for latent_idx in range(mu.shape[1]):
                latent_vals = mu[:, latent_idx]
                corr_coeff, _ = pearsonr(factor_vals, latent_vals)
                if np.abs(corr_coeff) > np.abs(best_corr):
                    best_corr = corr_coeff
                    best_latent_idx = latent_idx
            best_latent_indices.append(best_latent_idx)
            best_corr_coeffs.append(best_corr)

        z_reset = env.reset()['latent_observation']
        xy = np.linspace(z_min[best_latent_indices],
                         z_max[best_latent_indices], grid_size)
        goals = []
        for y in xy[:, 1]:
            g = []
            for x in xy[:, 0]:
                goal = z_reset.copy()
                goal[best_latent_indices] = [x, y]
                g.append(goal)
            goals.append(g)
        goals = np.array(goals)

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(
        grid_size, grid_size,
        left=0.1, right=1, bottom=0.1, top=1, wspace=0.01, hspace=0.01)

    for i in range(goals.shape[0]):
        for j in range(goals.shape[1]):
            goal = goals[i, j]
            policy.reset()
            o = env.reset()

            images = []
            path_length = 0
            while path_length < max_path_length:
                new_obs = np.hstack((o['latent_observation'], goal))
                action = policy.get_action(new_obs)
                next_o, reward, done, env_info = env.step(action[0])
                images.append(next_o['image_observation'])

                if render:
                    env.render()
                if done:
                    break

                o = next_o
                path_length += 1

            image = process_image_trajectory(images)

            ax = fig.add_subplot(gs[i, j])
            ax.imshow(image)
            ax.set_adjustable('box')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == goals.shape[0] - 1:
                ax.set_xlabel(f'{goal[0]:.2f}')
            if j == 0:
                ax.set_ylabel(f'{goal[1]:.2f}')

    gs_exp_id = gs_path.split('/')[-1]
    fig_path = os.path.join(
        output, f'{env_id}-{desired_goal_key}-gs{grid_size}-mpl{max_path_length}-{gs_exp_id}.png'
    )
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
