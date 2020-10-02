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
Generates a weakly-labelled dataset of observations.

Double-click pixel to set position.
Press {'r', 'y', 'g', 'b', 't'} to label object position.
Press 'n' to go to next image, or '-' to go to previous image.
Press 'c' to copy previous labels to the current image.
Press 'z' to clear all labels for the current image.
Press 'd' to toggle do_not_use flag.
Press 's' to save.
Press 'q' to exit & save labelled data.

python -m scripts.generate_hardware_datasets --input-paths output/robot_observations-2020-02-28-17:07:51.npz,output/robot_observations-2020-02-28-18:41:34.npz
"""

import os
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

import cv2
import click
import gym
import matplotlib.pyplot as plt
import numpy as np

from weakly_supervised_control.envs import register_all_envs
from weakly_supervised_control.envs.env_util import get_camera_fn
from weakly_supervised_control.envs.hardware_robot import concat_images
from weakly_supervised_control.disentanglement.np_data import NpGroundTruthData


KEY_TO_LABEL = {
    ord('r'): 'obj_pos_red',
    ord('b'): 'obj_pos_blue',
    ord('g'): 'obj_pos_green',
    ord('y'): 'obj_pos_yellow',
    ord('t'): 'obj_pos_purple',
}

LABEL_TO_COLOR = {  # (b, g, r)
    'obj_pos_red': (0, 0, 255),
    'obj_pos_blue': (255, 0, 0),
    'obj_pos_green': (0, 153, 0),
    'obj_pos_yellow': (0, 255, 255),
    'obj_pos_purple': (255, 0, 255),
}


def plot_sample(dset: NpGroundTruthData, save_path: str = None):
    x1, x2, y = dset.sample_rank_pair(1)
    x1 = concat_images(x1[0])
    x2 = concat_images(x2[0])

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(x1)
    axes[1].imshow(x2)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def get_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        param['position'] = (x, y)
        print(param)


def save_dset(data, factors, factor_names, save_prefix: str = ""):
    dset = NpGroundTruthData(
        data, factors, factor_names=factor_names)
    output_prefix = save_prefix + f'-n{len(data)}'
    dset.save(output_prefix + '.npz')
    plot_sample(dset, save_path=output_prefix + '.png')


def show_image(observations: Dict,
               idx: int,
               print_labels: bool = False,
               width: int = 1000,
               height: int = 1000):
    if idx >= len(observations):
        print(
            f"Index {idx} greater than number of observations {len(observations)}")
        return
    if print_labels:
        print(f'Image {idx} (out of {len(observations)}):')
        print('do_not_use:', observations[idx].get('do_not_use', False))

    images = observations[idx]['image_observation'].copy()
    images = concat_images(images, resize_shape=(1000, 1000))
    for label in KEY_TO_LABEL.values():
        position = observations[idx].get(label, None)
        if print_labels:
            print(f"{label}: {position}")

        color = LABEL_TO_COLOR[label]
        cv2.circle(images, position, 20, color, -1)
    cv2.imshow('image', images)


@click.command()
@click.option('--input-paths', type=str, help="Comma-separated list of paths")
@click.option('--test-data-size', type=int, default=256, help="Comma-separated list of paths")
def main(input_paths: str, test_data_size: int):
    # Read observations.
    input_paths = input_paths.split(',')
    observations = []
    for input_path in input_paths:
        data = np.load(input_path, allow_pickle=True)
        observations += data['observations'].tolist()
    print(
        f'Loaded {len(observations)} observations from {len(input_paths)} files.')
    output_path = 'output/robot_observations-' + \
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    param = {}
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_mouse_position, param)

    idx = 0
    prev_idx = None
    while idx < len(observations):
        show_image(observations, idx)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):    # Exit
            break
        elif key == ord('s'):  # Save
            with open(output_path + '.npz', 'wb') as f:
                np.savez(f, observations=observations)
            print(f"Saved to {output_path}.npz")
        elif key == ord('x'):  # Delete current image
            print(f"Deleting labels for image {idx}")
            for label in KEY_TO_LABEL.values():
                if label in observations[idx]:
                    del observations[idx][label]
            show_image(observations, idx, print_labels=True)
        elif key == ord('n'):  # Next image
            prev_idx = idx
            idx += 1
            show_image(observations, idx, print_labels=True)
        elif key == ord('-'):  # Previous image
            prev_idx = idx
            idx -= 1
            show_image(observations, idx, print_labels=True)
        elif key == ord('c'):  # Copy previous labels
            if prev_idx is None:
                print(f"Cannot copy previous labels for idx {idx}")
                continue
            for label in KEY_TO_LABEL.values():
                if label in observations[prev_idx]:
                    observations[idx][label] = observations[prev_idx][label]
            show_image(observations, idx, print_labels=True)
        elif key == ord('d'):  # Do not use
            observations[idx]['do_not_use'] = not observations[idx].get(
                'do_not_use', False)
            show_image(observations, idx, print_labels=True)
        elif key in KEY_TO_LABEL.keys():  # Label image
            label = KEY_TO_LABEL[key]
            observations[idx][label] = param['position']
            show_image(observations, idx, print_labels=True)

    # Save labelled data.
    with open(output_path + '.npz', 'wb') as f:
        np.savez(f, observations=observations)
    print(f"Saved to {output_path}.npz")

    # Create dataset.
    data = []
    factors = []
    for i, o in enumerate(observations):
        # Skip images with occlusion.
        if o.get('do_not_use', False):
            continue

        y = o['end_effector_pos'].tolist()
        for label in KEY_TO_LABEL.values():
            if label in o:
                pos = o[label]
            else:
                # Skip images with incomplete labels
                print(f"Skipping image {i}: {o}")
                y = None
                break
            y += pos
        if y is not None:
            obs = o['image_observation'] / 255.0
            data.append(obs)
            factors.append(y)
    data = np.array(data)
    factors = np.array(factors)

    factor_names = ['hand_pos_x', 'hand_pos_y', 'hand_pos_z']
    for label in KEY_TO_LABEL.values():
        factor_names += [label + '_x', label + '_y']

    # Split into train/test sets.
    indices = np.random.permutation(len(data))
    test_indices = indices[:test_data_size]
    train_indices = indices[test_data_size:]
    save_dset(data[train_indices], factors[train_indices],
              factor_names, save_prefix=output_path)
    save_dset(data[test_indices], factors[test_indices],
              factor_names, save_prefix=output_path)


if __name__ == '__main__':
    main()
