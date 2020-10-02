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
Opening a door with a Sawyer arm.

To test:
python -m weakly_supervised_control.envs.sawyer_door
"""
from typing import Dict
import os

from gym import spaces
import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv


class SawyerDoorGoalEnv(SawyerDoorHookEnv):
    def __init__(self, **sawyer_xyz_kwargs):
        from multiworld.envs import env_util
        old_asset_dir = env_util.ENV_ASSET_DIR
        env_util.ENV_ASSET_DIR = os.path.join(
            os.path.dirname(__file__), 'assets')
        super().__init__(**sawyer_xyz_kwargs)
        env_util.ENV_ASSET_DIR = old_asset_dir

        self._door_start_id = self.model.site_name2id('door_start')
        self._door_end_id = self.model.site_name2id('door_end')

    @property
    def factor_names(self):
        return ['hand_x', 'hand_y', 'hand_z', 'door_angle']

    @property
    def door_endpoints(self):
        door_start = self.data.get_site_xpos('door_start')
        door_end = self.data.get_site_xpos('door_end')
        return [door_start, door_end]

    def set_to_goal_angle(self, angle):
        self.data.set_joint_qpos('doorjoint', angle)
        self.data.set_joint_qvel('doorjoint', angle)
        self.sim.forward()

    def set_to_goal_pos(self, xyz, error_tol: float = 1e-2, max_iters: int = 1000):
        self.data.set_mocap_pos('mocap', np.array(xyz))
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        u = self.data.ctrl.copy()
        u[:-1] = 0

        # Step the simulation until the hand is close enough to the desired pos.
        error = 0
        for sim_iter in range(max_iters // self.frame_skip):
            self.do_simulation(u, self.frame_skip)
            cur_hand_pos = self.get_endeff_pos()
            error = np.linalg.norm(xyz - cur_hand_pos)
            if error < error_tol:
                break
        # print(f'Took {sim_iter * self.frame_skip} (error={error}) to converge')

    def set_to_goal(self, goal: Dict[str, np.ndarray]):
        """
        This function can fail due to mocap imprecision or impossible object
        positions.
        """
        state_goal = goal['state_desired_goal']
        assert state_goal.shape == (4,), state_goal.shape

        self.set_to_goal_pos(state_goal[:3])
        self.set_to_goal_angle(state_goal[-1])

    def sample_goals(self, batch_size, close_threshold: float = 0.05):
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )
        # This only works for 2D control
        # goals[:, 2] = self.fixed_hand_z

        for goal in goals:
            hand_pos = goal[:3]
            door_angle = goal[3]
            self.set_to_goal_angle(door_angle)

            door_start, door_end = self.door_endpoints
            door_vec = door_end - door_start
            door_vec /= np.linalg.norm(door_vec)

            door_to_hand = hand_pos - door_start
            door_to_hand[2] = 0

            proj = np.dot(door_vec, door_to_hand) * door_vec
            normal_vec = door_to_hand - proj
            length = np.linalg.norm(normal_vec)

            # If the arm is inside the door, move it outside.
            if normal_vec[1] > 0:
                hand_pos = hand_pos - 2 * normal_vec
                normal_vec *= -1

            if length < close_threshold:
                perturb = normal_vec * 2 * close_threshold / length
                hand_pos += perturb

            goal[:3] = hand_pos

        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def _set_debug_pos(self, pos, i):
        marker_id = self.model.geom_name2id(f'marker{i}')
        self.model.geom_pos[marker_id, :] = pos
        self.sim.forward()


if __name__ == '__main__':
    import gym
    from weakly_supervised_control.envs import register_all_envs
    from weakly_supervised_control.envs.env_wrapper import MujocoSceneWrapper
    from weakly_supervised_control.envs.multiworld.envs.mujoco import cameras

    register_all_envs()
    env = gym.make('SawyerDoorGoalEnv-v1')

    env = MujocoSceneWrapper(env)
    for e in range(20):
        env.reset()
        env.initialize_camera(cameras.sawyer_pick_and_place_camera)

        goal = env.sample_goals(1)
        goal = {k: v[0] for k, v in goal.items()}
        env.set_to_goal(goal)

        for _ in range(100):
            env.render()
