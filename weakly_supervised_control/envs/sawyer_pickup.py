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
Picking up an object with a Sawyer arm.

To run test:
python -m weakly_supervised_control.envs.sawyer_pickup
"""
from gym import spaces
import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnvYZ


class SawyerPickupGoalEnv(SawyerPickAndPlaceEnvYZ):
    # The y,z offset of the ball relative to the end-effector.
    BALL_IN_HAND_OFFSET = np.array([0, 0.00, -0.04])

    def __init__(self, grab_object_probability: float = 0.2, **kwargs):
        self.quick_init(locals())

        # Probability of sampling object goal position below the hand.
        self.grab_object_probability = grab_object_probability

        self._hand_space = None
        self._object_space = None
        super().__init__(**kwargs)

    @property
    def factor_names(self):
        return ['hand_x', 'hand_y', 'hand_z', 'obj_x', 'obj_y', 'obj_z']

    @property
    def object_space(self):
        if self._object_space is None:
            obj_space_low = self.hand_and_obj_space.low[[3, 4, 5]]
            obj_space_high = self.hand_and_obj_space.high[[3, 4, 5]]
            self._object_space = spaces.Box(
                low=obj_space_low,
                high=obj_space_high)
        return self._object_space

    @property
    def hand_space_higher_z(self):
        if self._hand_space is None:
            hand_space_low = self.hand_and_obj_space.low[[0, 1, 2]]
            hand_space_high = self.hand_and_obj_space.high[[0, 1, 2]]
            self._hand_space = spaces.Box(
                low=hand_space_low,
                high=hand_space_high)
        return self._hand_space

    def sample_single_goal(self):
        hand_pos = self.hand_space_higher_z.sample()
        hand_pos[0] = 0

        if self.np_random.rand() < self.grab_object_probability:
            # Place the object in the hand.
            obj_pos = hand_pos + self.BALL_IN_HAND_OFFSET
        else:
            # Place the object on the table.
            obj_pos = self.object_space.sample()
            obj_pos[2] = self.obj_init_z
        obj_pos[0] = 0

        return np.concatenate([hand_pos, obj_pos])

    def sample_goals(self, batch_size: int):
        goals = np.stack([self.sample_single_goal()
                          for _ in range(batch_size)])
        return {
            'state_desired_goal': goals,
            'desired_goal': goals,
            'proprio_desired_goal': goals[:, :3]
        }

    def set_to_goal(self, goal):
        """
        This function can fail due to mocap imprecision or impossible object
        positions.
        """
        state_goal = goal['state_desired_goal']
        assert state_goal.shape == (6,), state_goal.shape

        # Move the hand to the desired position.
        desired_hand_pos = state_goal[:3]
        self.data.set_mocap_pos('mocap', desired_hand_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        # Step the simulation until the hand is close enough to the desired pos.
        error = 0
        for _ in range(1000):
            self.do_simulation(np.array([-1]))
            cur_hand_pos = self.get_endeff_pos()
            error = np.linalg.norm(desired_hand_pos - cur_hand_pos)
            if error < 3e-3:
                break
        # Set the object position.
        self._set_obj_xyz(np.array(state_goal[3:]))
        self.sim.forward()


if __name__ == '__main__':
    from weakly_supervised_control.envs.env_wrapper import MujocoSceneWrapper

    env = SawyerPickupGoalEnv()
    env = MujocoSceneWrapper(env)
    for e in range(20):
        env.reset()

        goal = env.sample_goals(1)
        goal = {k: v[0] for k, v in goal.items()}
        env.set_to_goal(goal)

        for _ in range(50):
            env.render()
