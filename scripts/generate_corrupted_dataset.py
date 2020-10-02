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

python -m scripts.generate_corrupted_dataset --input gs://weakly-supervised-control/datasets/SawyerPushRandomLightsEnv-v1-n256-hand_x-hand_y-obj_x-obj_y-light.npz --noise 0.1
python -m scripts.generate_corrupted_dataset --input gs://weakly-supervised-control/datasets/SawyerPushRandomLightsEnv-v1-n256-hand_x-hand_y-obj_x-obj_y-light.npz --noise 0.05
python -m scripts.generate_corrupted_dataset --input gs://weakly-supervised-control/datasets/SawyerPush2PucksRandomLightsEnv-v1-n512-hand_x-hand_y-obj1_x-obj1_y-obj2_x-obj2_y-light.npz --noise 0.05
python -m scripts.generate_corrupted_dataset --input gs://weakly-supervised-control/datasets/SawyerPush3PucksRandomLightsEnv-v1-n512-hand_x-hand_y-obj1_x-obj1_y-obj2_x-obj2_y-light.npz --noise 0.05
"""

import os
from typing import Any, Callable, Dict, List, Optional

import click
import gym
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.io import gfile

from weakly_supervised_control.envs import register_all_envs
from weakly_supervised_control.experiment_utils import load_dset
# from weakly_supervised_control.disentanglement.np_data import NpGroundTruthData


@click.command()
@click.option('--input', type=str, help="Input dataset path")
@click.option('--noise', type=float, default=0.05, help="Probability of corrupting a factor label")
@click.option('--num-output', type=int, default=5, help="Number of corrupted datasets to create")
def main(input: str, noise: float, num_output: int):
    dset = load_dset(input)

    for n in range(num_output):
        num_corrupted_labels = 0
        for i in range(dset.size):
            # Corrupt factor labels with 0.05 probability.
            one_hot = np.random.choice([False, True], size=5, p=[1 - noise, noise])

            for j, x in enumerate(one_hot):
                if x:
                    fake_factor_value = np.random.uniform(0, 1)
                    dset.factors[i, j] = fake_factor_value
                    num_corrupted_labels += 1
        print(f'Corrupted {num_corrupted_labels}/{dset.size} labels.')

        # Save to file.
        temp_file = f'/tmp/generate_corrupted_dataset_output-{n}.npz'
        dset.save(temp_file)
        print(f'Saved to: {temp_file}')

        output_prefix = input.split('.npz')[0] + f"-noise{noise}-seed{n}"
        gfile.copy(temp_file, f'{output_prefix}.npz')
        gfile.remove(temp_file)
        print(f'Saved to: {output_prefix}.npz')

if __name__ == '__main__':
    main()
