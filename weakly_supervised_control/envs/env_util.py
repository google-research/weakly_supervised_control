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
from typing import Tuple
import gym
import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
#from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEnv, SawyerPushAndReachXYEasyEnv
from multiworld.envs.mujoco import cameras

from weakly_supervised_control.envs.sawyer_push import SawyerPushGoalXYEasyEnv
from weakly_supervised_control.envs.sawyer_pickup import SawyerPickupGoalEnv
from weakly_supervised_control.envs.sawyer_door import SawyerDoorGoalEnv

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEasyEnv

def get_camera_fn(env: gym.Env):
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped
    if isinstance(env, SawyerPushGoalXYEasyEnv) or isinstance(env, SawyerPushAndReachXYEasyEnv):
        return cameras.sawyer_init_camera_zoomed_in
    elif isinstance(env, SawyerPickupGoalEnv):
        return cameras.sawyer_pick_and_place_camera
    elif isinstance(env, SawyerDoorGoalEnv):
        return cameras.sawyer_door_env_camera_v0
    else:
        return None


def normalize_image(image: np.ndarray, dtype=np.float64) -> np.ndarray:
    assert image.dtype == np.uint8
    return dtype(image) / 255.0


def unormalize_image(image: np.ndarray) -> np.ndarray:
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)


def concat_images(images: np.ndarray,
                  resize_shape: Tuple[int, int] = None,
                  concat_horizontal: bool = True):
    """Concatenates N images into a single image.
    Args:
      images: An RGB array of shape (N, width, height, 3)
      resize_shape: (width, height)
      concat_horizontal: If true, returns an image of size (N * width, height).
                         Otherwise returns an image of size (width, N * height).
    """
    assert len(images.shape) == 4, images.shape
    assert images.shape[3] == 3, images.shape
    if resize_shape:
        images = np.array([cv2.resize(image, resize_shape)
                           for image in images])
    if concat_horizontal:
        stacked_images = np.concatenate(images, axis=1)
    else:
        stacked_images = np.concatenate(images, axis=0)
    return stacked_images
