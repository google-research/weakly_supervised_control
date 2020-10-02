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
"""
Generates a dataset of observations from a multiworld environment.

python -m scripts.generate_sawyer_datasets --env-id SawyerPushRandomLightsEnv-v1

"""

import os
from typing import Any, Callable, Dict, List, Optional

import click
import gym
import matplotlib.pyplot as plt
import numpy as np

from weakly_supervised_control.envs import register_all_envs
from weakly_supervised_control.envs.env_util import get_camera_fn
from weakly_supervised_control.disentanglement.np_data import NpGroundTruthData


def plot_sample(dset: NpGroundTruthData, save_path: str = None):
    x1, x2, y = dset.sample_rank_pair(1)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x1[0])
    axes[1].imshow(x2[0])

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


@click.command()
@click.option('--env-id', default="SawyerPickupEnv-v1", type=str)
@click.option('--output', default="/tmp/weakly_supervised_control/datasets", type=str)
def main(env_id: str, output: str):
    # Create the environment.
    register_all_envs()
    env = gym.make(env_id)
    env.seed(0)
    env.reset()
    env.initialize_camera(get_camera_fn(env))

    os.makedirs(output, exist_ok=True)
    for n in [16, 32, 64, 128, 256, 512]:
        # Generate dataset.
        dset = NpGroundTruthData.sample_env_goals(env, num_samples=n, image_size=1000)

        output_prefix = os.path.join(output, f'{env_id}-n{n}')
        dset.save(output_prefix + '.npz')
        print(f'Saved to: {output_prefix}.npz')
        plot_sample(dset, save_path=output_prefix + '.png')


if __name__ == '__main__':
    main()
