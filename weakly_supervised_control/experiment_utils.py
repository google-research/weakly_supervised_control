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
import importlib
import json
import os
import re

import cv2
import gym
from gym.spaces import Box
import numpy as np
import pickle
from tensorflow.io import gfile
import torch

import multiworld
from rlkit.core import logger
from rlkit.torch.networks import FlattenMlp

from weakly_supervised_control.disentanglement.model import DisentanglementModel
from weakly_supervised_control.disentanglement.np_data import NpGroundTruthData
from weakly_supervised_control.envs.env_util import get_camera_fn
from weakly_supervised_control.envs.image_env import ImageEnv
from weakly_supervised_control.vae.conv_vae import VAE
from weakly_supervised_control.vae.vae_trainer import VAETrainer
from weakly_supervised_control.envs.disentangled_env import DisentangledEnv


def disable_tensorflow_gpu() -> None:
    """Disables CUDA for Tensorflow."""
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # This is necessary to prevent tensorflow from using the GPU.
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    if cuda_devices is None:
        del os.environ['CUDA_VISIBLE_DEVICES']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices


def _replace_paths(d):
    """Replace string paths with function pointers or absolute paths."""
    for key in d.keys():
        val = d[key]
        if isinstance(val, dict):
            _replace_paths(val)
        else:
            if isinstance(val, str):
                if val.startswith('multiworld/'):
                    # Replace val with a file path (str)
                    d[key] = os.path.join(
                        os.path.dirname(
                            os.path.dirname(multiworld.__file__)),
                        val
                    )
                elif val.startswith('rlkit.') or val.startswith('multiworld.'):
                    # Replace val with pointer to a function
                    items = val.split('.')
                    module = importlib.import_module('.'.join(items[:-1]))
                    d[key] = getattr(module, items[-1])


def load_config(config_path: str) -> Dict:
    """Loads config from file."""
    with gfile.GFile(config_path, 'r') as f:
        variant = eval(f.read())

    _replace_paths(variant)

    return variant


def load_dset(dset_path: str, stack_images: bool = False) -> NpGroundTruthData:
    dset = NpGroundTruthData.load(dset_path, stack_images=stack_images)

    print('Loaded dataset: {}'.format(dset_path))
    print('Data shape: {}'.format(dset.observation_shape))
    print('{} factors of sizes {}'.format(
        dset.num_factors,  dset.factors_num_values))
    return dset


def load_disentanglement_model(
        dset: NpGroundTruthData,
        factors: str = None,
        model_kwargs: Dict = {},
        factor_indices: Tuple[int] = None,
        model_path: str = None) -> Tuple[DisentanglementModel, gym.spaces.Box]:
    if model_path is not None:
        assert factors is None
        factors = re.findall(r"(r=\d(,\d)+)", model_path)[0][0]

    model = DisentanglementModel(
        dset.observation_shape, dset.num_factors, factors=factors, **model_kwargs)
    
    if model_path is not None:
        model.load_checkpoint(model_path)

    if factor_indices is not None:
        data = dset.sample(dset.size)
        z_min, z_max = model.get_latent_range(data)
        z_min = np.array([
            z_min[i] for i in factor_indices])
        z_max = np.array([
            z_max[i] for i in factor_indices])
        space = Box(z_min, z_max, dtype=np.float32)
    else:
        space = None

    return model, space


def create_disentangled_env(
        env: gym.Env,
        vae: VAE,
        eval_dset: NpGroundTruthData = None,
        disentangled_env_kwargs: Dict = {}) -> ImageEnv:
    presampled_goals = None
    if eval_dset is not None:
        presampled_goals = {
            'image_desired_goal': eval_dset.data,
            'state_desired_goal': eval_dset.factors,
        }

    if hasattr(env.unwrapped, 'initialize_camera'):
        init_camera = get_camera_fn(env)
        env.unwrapped.initialize_camera(init_camera)

    image_env = ImageEnv(
        wrapped_env=env,
        x_shape=eval_dset.observation_shape,
        presampled_goals=presampled_goals,
    )

    return DisentangledEnv(image_env,
                           vae,
                           imsize=image_env.imsize,
                           presampled_goals=image_env._presampled_goals,
                           **disentangled_env_kwargs)


def load_experiment(experiment_dir: str, use_cpu: bool = False):
    # Load config file.
    with gfile.GFile(os.path.join(experiment_dir, 'variant.json'), 'r') as f:
        variant = json.load(f)
    _replace_paths(variant)

    # Load policy, VAE, and environment.
    with gfile.GFile(os.path.join(experiment_dir, 'params.pkl'), 'rb') as f:
        if use_cpu:
            params = torch.load(f, map_location=lambda storage, loc: storage)
        else:
            params = torch.load(f)

    with gfile.GFile(os.path.join(experiment_dir, 'vae.pkl'), 'rb') as f:
        vae = pickle.load(f)

    return variant, params, vae


def train_vae(
    vae: VAE,
    dset: NpGroundTruthData,
    num_epochs: int = 0,  # Do not pre-train by default
    save_period: int = 5,
    test_p: float = 0.1,  # data proportion to use for test
    vae_trainer_kwargs: Dict = {},
):
    logger.remove_tabular_output(
        'progress.csv', relative_to_snapshot_dir=True
    )
    logger.add_tabular_output(
        'vae_progress.csv', relative_to_snapshot_dir=True
    )

    # Flatten images.
    n = dset.data.shape[0]
    data = dset.data.transpose(0, 3, 2, 1)
    data = data.reshape((n, -1))

    # Un-normalize images.
    if data.dtype != np.uint8:
        assert np.min(data) >= 0.0
        assert np.max(data) <= 1.0
        data = (data * 255).astype(np.uint8)

    # Make sure factors are normalized.
    assert np.min(dset.factors) >= 0.0
    assert np.max(dset.factors) <= 1.0

    # Split into train and test set.
    test_size = int(n * test_p)
    test_data = data[:test_size]
    train_data = data[test_size:]
    train_factors = dset.factors[test_size:]
    test_factors = dset.factors[:test_size]

    logger.get_snapshot_dir()

    trainer = VAETrainer(vae, train_data, test_data,
                         train_factors=train_factors,
                         test_factors=test_factors,
                         **vae_trainer_kwargs)

    for epoch in range(num_epochs):
        should_save_imgs = (epoch % save_period == 0)
        trainer.train_epoch(epoch)
        trainer.test_epoch(epoch, save_reconstruction=should_save_imgs)
        if should_save_imgs:
            trainer.dump_samples(epoch)
        trainer.update_train_weights()

    logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
    logger.remove_tabular_output(
        'vae_progress.csv',
        relative_to_snapshot_dir=True,
    )
    logger.add_tabular_output(
        'progress.csv',
        relative_to_snapshot_dir=True,
    )

    return vae, trainer


def process_image_trajectory(
        images,
        object_color: Tuple[float, float, float] = (0, 0.392, 0.784),
        threshold: float = 1e-2,
        draw_color: Tuple[float, float, float] = (1, 1, 1),
        alpha: float = 0.6,
        draw_arrow: bool = False,
        arrow_length: float = 3,
        arrow_width: float = 3):
    """Draws a line over the tracked object_color along the image trajectory."""
    object_color = np.array(object_color)
    lower = np.maximum(0.0, object_color - threshold)
    upper = np.minimum(1.0, object_color + threshold)

    centroids = []
    for image in images:
        image = image.reshape((3, 48, 48)).transpose((2, 1, 0))
        mask = cv2.inRange(image, lower, upper)
        centroid = np.mean(list(zip(*np.nonzero(mask))), axis=0)
        centroids.append(centroid)

    draw_indices = set()

    def _to_index(p: Tuple[float, float]) -> Tuple[int, int]:
        assert p.shape == (2,), p
        return tuple(map(int, p))

    # Walk along the trajectory and collect pixels to draw.
    pos = centroids[0].copy()
    draw_indices.add(_to_index(pos))
    directions = []
    dest_index = 1
    while dest_index < len(centroids):
        dest_pos = centroids[dest_index]
        dest_vec = dest_pos - pos
        dest_distance = np.linalg.norm(dest_vec)
        if dest_distance < 1e-2:
            dest_index += 1
            continue
        dest_direction = dest_vec / dest_distance
        step = np.minimum(dest_distance, 0.5)
        pos += dest_direction * step
        draw_indices.add(_to_index(pos))
        directions.append(dest_direction)

    if draw_arrow and directions:
        average_dir = np.mean(directions[-10:], axis=0)
        average_dir /= np.linalg.norm(average_dir)
        assert average_dir.shape == (2,)

        def _add_line(src, dest, iters: int = 20):
            assert src.shape == (2,) and dest.shape == (2,), (src, dest)
            for i in range(iters):
                a = i / (iters - 1)
                v = (1. - a) * src + a * dest
                draw_indices.add(_to_index(v))

        # Add an arrow head.
        tri_base = centroids[-1] - average_dir * arrow_length
        ortho_vec = np.array(
            [average_dir[1], -average_dir[0]]) * arrow_width * 0.5
        tri_right = tri_base + ortho_vec
        tri_left = tri_base - ortho_vec
        # _add_line(tri_left, tri_right)
        _add_line(tri_left, centroids[-1])
        _add_line(tri_right, centroids[-1])

    # Draw on the last image.
    result = image.copy()
    draw_color = np.array(draw_color)
    for index in draw_indices:
        if any(i < 0 or i >= result.shape[0] for i in index):
            continue
        result[index] = (1 - alpha) * result[index] + alpha * draw_color

    return result


def get_grid_space():
    # TODO: Implement this.
    pass
