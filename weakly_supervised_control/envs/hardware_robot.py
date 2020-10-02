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
from multiworld.core.serializable import Serializable
from typing import Dict, List, Tuple
import cv2
import gym
from gym import spaces
import numpy as np
import time
from datetime import datetime

try:
    from juggler.libraries import robot_client
except:
    print("Failed to import juggler")

from weakly_supervised_control.envs.env_util import normalize_image, unormalize_image, concat_images

JOINT_POS_SPACE = np.array((2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05))
JOINT_VEL_SPACE = np.array((1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36))

# Franka settings
INIT_QUATERNION = np.array((0, 1, 0, 0))  # For Kuka, use (0, 0, 1, 0).
MIN_POS = np.array((0.355, -0.135, 0.248))#0.257845))
MAX_POS = np.array((0.545, 0.105, 0.257845))
MAX_VEL_ACTION = np.array((0.12, 0.12, 0.01))
MAX_VEL_PER_COMMAND = 0.04
COLLISION_DIST_THRESHOLD = 1e-2


def get_image_from_camera(camera) -> np.ndarray:
    """Reads an image from the camera."""
    assert camera is not None
    while True:
        try:
            _, image = camera.read()
            if len(image) > 0:
                break
        except:
            print(image)
    return image


class HardwareRobotEnv(gym.Env, Serializable):
    def __init__(self,
                 camera_ids: List[int] = [2, 0],
                 frequency: int = 2,
                 reset_blocks_freq: int = 5,
                 stack_images: bool = True):
        self.quick_init(locals())
        self.client = robot_client.make_robot_client_from_commandline()
        self.client.send_command(
            robot_client.RobotCommandBuilder().enable_motion())

        self.reset_blocks_freq = reset_blocks_freq
        self.stack_images = stack_images
        self.last_pos = np.zeros(3)
        self.timestep = 1. / frequency
        self.num_resets = 0

        # Initialize cameras.
        self.cameras = []
        if camera_ids:
            for camera_id in camera_ids:
                camera = cv2.VideoCapture(camera_id)
                assert camera is not None, camera_id
                self.cameras.append(camera)

        if self.stack_images:
            image_obs_shape = (48, 48, 3 * len(self.cameras))
        else:
            image_obs_shape = (len(self.cameras), 48, 48, 3)

        self.observation_space = spaces.Dict({
            'joint_pos': spaces.Box(
                low=-JOINT_POS_SPACE, high=JOINT_POS_SPACE
            ),
            'joint_vel': spaces.Box(
                low=-JOINT_VEL_SPACE, high=JOINT_VEL_SPACE,
            ),
            'end_effector_pos': spaces.Box(
                low=MIN_POS, high=MAX_POS,
            ),
            'end_effector_quaternion': spaces.Box(
                low=-1, high=1, shape=(4,), dtype=float
            ),
            'image_observation': spaces.Box(
                low=0, high=255, shape=image_obs_shape, dtype=int),
        })
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=float)

        command = robot_client.RobotCommandBuilder()
        command.set_part('arm').set_blocking().joint_position_move(
            (0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0))
        self.client.send_command(command)

        # command = robot_client.RobotCommandBuilder()
        # command.set_part('gripper').gripper_move(0.02, 1.0)
        # self.client.send_command(command)

        self.reset()

    def _get_arm_pos(self):
        state = self.client.get_state()

        joint_pos = np.asarray(state.part_states[
            'arm'].joint_position_sensed.data)
        joint_vel = np.asarray(state.part_states[
            'arm'].joint_velocity_sensed.data)

        t = state.part_states['arm'].pose_sensed.translation
        end_effector_pos = np.asarray([t.x, t.y, t.z])

        t = state.part_states['arm'].pose_sensed.rotation
        end_effector_quaternion = np.asarray([t.w, t.x, t.y, t.z])

        return joint_pos, joint_vel, end_effector_pos, end_effector_quaternion

    def get_image(self, width: int = 48, height: int = 48) -> np.ndarray:
        """Returns a list of images."""
        images = []
        for camera in self.cameras:
            if camera is None:
                print("Warning: Camera is not set")
                image = np.zeros((width, height, 3))
            else:
                image = get_image_from_camera(camera)
                image = cv2.resize(image, (width, height))
            images.append(image)
        images = np.array(images)

        if self.stack_images:
            # Images should be of shape (num_images, w, h, 3).
            assert len(images.shape) == 4, images.shape
            assert images.shape[-1] == 3, images.shape
            # Reshape to (w, h, 3 * num_images).
            images = np.concatenate(images, axis=2)
        return images

    def render(self, width: int = 500, height: int = 500) -> None:
        if not self.cameras:
            return
        images = self.get_image(width, height)  # Do not resize
        images = concat_images(images)
        cv2.imshow('image', images)
        cv2.waitKey(1)

    def _get_obs(self):
        # Get arm pos
        joint_pos, joint_vel, end_effector_pos, end_effector_quaternion = self._get_arm_pos()
        obs = {
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'end_effector_pos': end_effector_pos,
            'end_effector_quaternion': end_effector_quaternion,
        }

        # Get image
        if self.cameras:
            images = self.get_image()
            obs['image_observation'] = images

        return obs

    def _move_arm_fast(self, pos: np.ndarray):
        command = robot_client.RobotCommandBuilder()
        command.set_part('arm').set_blocking(
        ).cartesian_position_move(pos, INIT_QUATERNION)
        self.client.send_command(command)

    def _move_arm(self, xyz: np.ndarray, max_vel_per_command: float = None, blocking: bool = False):
        if max_vel_per_command is None:
            max_vel_per_command = MAX_VEL_PER_COMMAND
        xyz = np.clip(xyz, MIN_POS, MAX_POS)

        _, _, self.last_pos, _ = self._get_arm_pos()
        vel = xyz - self.last_pos
        num_commands = max(
            int(np.ceil(np.max(np.abs(vel)) / max_vel_per_command)), 1)
        vel_per_command = vel / num_commands
        #print(f'xyz: {xyz}, vel: {vel}, num_commands: {num_commands}, vel_per_command: {vel_per_command}')

        command = robot_client.RobotCommandBuilder()
        for i in range(num_commands):
            pos = self.last_pos + vel_per_command
            pos = np.clip(pos, MIN_POS, MAX_POS)

            if blocking:
                command.set_part('arm').set_blocking(
                ).cartesian_position_move(pos, INIT_QUATERNION)
                self.client.send_command(command)
            else:
                command.set_part('arm').cartesian_position_move(
                    pos, INIT_QUATERNION)
                self.client.send_command(command)
                time.sleep(self.timestep)

            _, _, self.last_pos, _ = self._get_arm_pos()

            # Check for collision.
            pos_diff = np.linalg.norm(self.last_pos - pos)
            if pos_diff > COLLISION_DIST_THRESHOLD:
                #print(f"Collision detected (actual: {self.last_pos}, expected: {pos}, diff: {pos_diff}). Clipping action after {i+1}/{num_commands} commands.")
                break

    def step(self, action: np.ndarray):
        assert action.shape == (3,), action.shape
        action = action * MAX_VEL_ACTION
        assert np.all(action <= MAX_VEL_ACTION), action

        # Execute action
        self._move_arm(self.last_pos + action)

        obs = self._get_obs()
        return obs, 0, False, {}

    def _reset_arm(self):
        """Resets arm position to initial position."""
        command = robot_client.RobotCommandBuilder()
        init_pos = np.array(0.5 * (MIN_POS + MAX_POS))
        command.set_part('arm').set_blocking().cartesian_position_move(
            init_pos, INIT_QUATERNION)
        self.client.send_command(command)

        # Update last position.
        self.last_pos = init_pos.copy()

    def _reset_blocks(self):
        """Sweeps arm around the bin to randomize block positions."""
        center_pos_low = np.array(0.5 * (MIN_POS + MAX_POS))
        center_pos_low[2] = MIN_POS[2]

        center_pos_high = np.array(0.5 * (MIN_POS + MAX_POS))
        center_pos_high[2] = MAX_POS[2] + 0.05

        x_min = MIN_POS[0]
        x_max = MAX_POS[0]
        y_min = MIN_POS[1]
        y_max = MAX_POS[1]
        z_min = MIN_POS[2]
        z_max = MAX_POS[2] + 0.05
        slow_vel = 0.05
        fast_vel = 0.1
        delta_y = (y_max - y_min) / 2
        delta_x = (x_max - x_min) / 2

        # Move arm along the edges of the bin, pushing blocks to the center.
        #   1    2
        #   4    3
        command = robot_client.RobotCommandBuilder()
        print("Moving arm from 1 to 2...")
        i = 0
        y = y_min + delta_y * i
        while y < y_max:
            command.set_part('arm').set_blocking().cartesian_position_move(
                np.array([x_min, y, z_max]), INIT_QUATERNION)
            self.client.send_command(command)

            self._move_arm(np.array([x_min, y, z_min]),
                           slow_vel, blocking=True)
            self._move_arm(center_pos_low + [-0.05, 0, 0], fast_vel, blocking=True)

            # command.set_part('arm').set_blocking().cartesian_position_move(
            #     center_pos_high, INIT_QUATERNION)
            # self.client.send_command(command)
            i += 1
            y = y_min + delta_y * i

        print("Moving arm from 2 to 3...")
        i = 0
        x = x_min + delta_x * i
        while x < x_max:
            command.set_part('arm').set_blocking().cartesian_position_move(
                np.array([x, y_max, z_max]), INIT_QUATERNION)
            self.client.send_command(command)

            self._move_arm(np.array([x, y_max, z_min]),
                           slow_vel, blocking=True)
            self._move_arm(center_pos_low + [0, 0.05, 0], fast_vel, blocking=True)

            # command.set_part('arm').set_blocking().cartesian_position_move(
            #     center_pos_high, INIT_QUATERNION)
            # self.client.send_command(command)
            i += 1
            x = x_min + delta_x * i

        print("Moving arm from 3 to 4...")
        i = 0
        y = y_max - delta_y * i
        while y > y_min:
            command.set_part('arm').set_blocking().cartesian_position_move(
                np.array([x_max, y, z_max]), INIT_QUATERNION)
            self.client.send_command(command)

            self._move_arm(np.array([x_max, y, z_min]),
                           slow_vel, blocking=True)
            self._move_arm(center_pos_low + [0.05, 0, 0], fast_vel, blocking=True)

            # command.set_part('arm').set_blocking().cartesian_position_move(
            #     center_pos_high, INIT_QUATERNION)
            # self.client.send_command(command)
            i += 1
            y = y_max - delta_y * i

        print("Moving arm from 4 to 1...")
        i = 0
        x = x_max - delta_x * i
        while x > x_min:
            command.set_part('arm').set_blocking().cartesian_position_move(
                np.array([x, y_min, z_max]), INIT_QUATERNION)
            self.client.send_command(command)

            self._move_arm(np.array([x, y_min, z_min]),
                           slow_vel, blocking=True)
            self._move_arm(center_pos_low + [-0.05, 0, 0], fast_vel, blocking=True)

            # command.set_part('arm').set_blocking().cartesian_position_move(
            #     center_pos_high, INIT_QUATERNION)
            # self.client.send_command(command)
            i += 1
            x = x_min - delta_x * i

        command.set_part('arm').set_blocking().cartesian_position_move(
            np.array([x_min, y_min, z_max]), INIT_QUATERNION)
        self.client.send_command(command)

        self._move_arm(np.array([x_min, y_min, z_min]),
                       slow_vel, blocking=True)
        self._move_arm(center_pos_low, fast_vel, blocking=True)

        # command.set_part('arm').set_blocking().cartesian_position_move(
        #     center_pos_high, INIT_QUATERNION)
        # self.client.send_command(command)

        # self._move_arm(np.array([x_min, y_min, z_min]),
        #                fast_vel, blocking=True)
        # self._move_arm(np.array([x_min, y_max, z_min]),
        #                fast_vel, blocking=True)
        # self._move_arm(np.array([x_max, y_max, z_min]),
        #                fast_vel, blocking=True)
        # self._move_arm(np.array([x_max, y_min, z_min]),
        #                fast_vel, blocking=True)
        # self._move_arm(np.array([x_min, y_min, z_min]),
        #                fast_vel, blocking=True)
        # self._move_arm(np.array([x_min, y_max, z_min]),
        #                fast_vel, blocking=True)

        # Move arm back to reset position
        print("Reset pos")
        init_pos = np.array(0.5 * (MIN_POS + MAX_POS))
        self._move_arm(init_pos, fast_vel, blocking=True)

        # Update last position.
        self.last_pos = init_pos.copy()

    def reset(self):
        if (self.num_resets == 0 or (self.reset_blocks_freq is not None and self.num_resets % self.reset_blocks_freq == 0)):
            print("Resetting blocks...")
            self._reset_blocks()
            print("Finished resetting blocks.")
        else:
            self._reset_arm()
        self.num_resets += 1

        obs = self._get_obs()
        return obs

    def set_goal(self, goal: Dict[str, np.ndarray]):
        pass

    def set_to_goal(self, goal: Dict[str, np.ndarray]):
        pass

    def compute_rewards(self, actions: np.ndarray, obs: np.ndarray):
        return np.zeros(actions.shape[0])

    def get_diagnostics(self, paths, **kwargs):
        return {}


if __name__ == '__main__':
    EXIT_KEY = 'q'
    save_freq = 1
    num_steps_per_key_input = 10000

    env = HardwareRobotEnv(frequency=2, camera_ids=[0, 2], stack_images=False)
    env.reset()

    t = 0
    observations = []
    key_input = None
    while key_input != EXIT_KEY:
        for _ in range(num_steps_per_key_input * save_freq):
            # Take one env step.
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)

            if t % save_freq == 0:
                #print(f't: {t}')
                # print(obs)
                env.render()
                observations.append(obs)

            t += 1

        key_input = input("Enter q to exit, or any other key to continue.")

    observations = np.stack(observations)
    output_npz = 'output/robot_observations-' + \
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.npz'
    with open(output_npz, 'wb') as f:
        np.savez(f, observations=observations)
    print(f"Saved to {output_npz}")
