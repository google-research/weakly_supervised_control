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

import os
import sys

from gym.envs.registration import register

def fix_multiworld_path():
    """Fixes multiworld import paths."""
    path = os.path.join(os.path.dirname(__file__),
                        '../../dependencies/multiworld')
    path = os.path.normpath(path)
    sys.path.append(path)

fix_multiworld_path()

from multiworld import register_all_envs as multiworld_register_all_envs
from multiworld.envs.mujoco.sawyer_xyz import sawyer_push_nips
from weakly_supervised_control.envs.env_wrapper import (
    create_wrapped_env,
    MujocoRandomLightsWrapper,
    MujocoRandomColorWrapper,
)
from weakly_supervised_control.envs.hardware_robot import HardwareRobotEnv
from weakly_supervised_control.envs.sawyer_pickup import SawyerPickupGoalEnv
from weakly_supervised_control.envs.sawyer_door import SawyerDoorGoalEnv
from weakly_supervised_control.envs.sawyer_push import (SawyerPushGoalXYEasyEnv,
                                                        SawyerPushGoalHarderEnv)

TABLE_COLORS = [
    (.6, .6, .5, 1),
    (1., .6, .5, 1),
    (.1, 0., .1, 1),
    (.6, 1., 1., 1),
    (1., 1., .5, 1),
]

BALL_COLORS = [
    (1., 0, 0, 1),
    (0, 1., 0, 1),
    (0, 0, 1., 1),
]


def register_all_envs():
    multiworld_register_all_envs()

    register(id='HardwareRobotEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=HardwareRobotEnv,
                 env_kwargs={},
                 wrappers=[
                 ],
             ))

    # Sawyer push environments
    # ========================

    PUSH_SHARED_KWARGS = dict(
        force_puck_in_goal_space=False,
        mocap_low=(-0.1, 0.55, 0.0),
        mocap_high=(0.1, 0.65, 0.5),
        hand_goal_low=(-0.1, 0.55),
        hand_goal_high=(0.1, 0.65),
        puck_goal_low=(-0.15, 0.5),
        puck_goal_high=(0.15, 0.7),
        hide_goal=True,
        reward_info=dict(type="state_distance", ),
    )

    register(id='SawyerPushRandomLightsEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=sawyer_push_nips.SawyerPushAndReachXYEasyEnv,
                 env_kwargs=PUSH_SHARED_KWARGS,
                 wrappers=[
                     (MujocoRandomLightsWrapper, dict()),
                 ],
             ))
    register(id='SawyerPush2PucksEnv-v1',
             entry_point=SawyerPushGoalXYEasyEnv,
             kwargs=dict(
                 puck_count=2,
                 **PUSH_SHARED_KWARGS,
             ))
    register(id='SawyerPush3PucksEnv-v1',
             entry_point=SawyerPushGoalXYEasyEnv,
             kwargs=dict(
                 puck_count=3,
                 **PUSH_SHARED_KWARGS,
             ))
    register(id='SawyerPush2PucksRandomLightsEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=SawyerPushGoalXYEasyEnv,
                 env_kwargs=dict(
                     puck_count=2,
                     **PUSH_SHARED_KWARGS,
                 ),
                 wrappers=[
                     (MujocoRandomLightsWrapper, dict()),
                 ],
             ))
    register(id='SawyerPush3PucksRandomLightsEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=SawyerPushGoalXYEasyEnv,
                 env_kwargs=dict(
                     puck_count=3,
                     **PUSH_SHARED_KWARGS,
                 ),
                 wrappers=[
                     (MujocoRandomLightsWrapper, dict()),
                 ],
             ))

    # Sawyer pickup environments
    # ==========================

    PICKUP_SHARED_KWARGS = dict(
        hand_low=(-0.1, 0.55, 0.05),
        hand_high=(0.0, 0.65, 0.13),
        action_scale=0.02,
        hide_goal_markers=True,
        num_goals_presampled=1000,
        p_obj_in_hand=.75,
    )

    register(
        id='SawyerPickupGoalEnv-v1',
        entry_point='weakly_supervised_control.envs.sawyer_pickup:SawyerPickupGoalEnv',
        kwargs=PICKUP_SHARED_KWARGS,
    )
    register(id='SawyerPickupRandomLightsEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=SawyerPickupGoalEnv,
                 env_kwargs=PICKUP_SHARED_KWARGS,
                 wrappers=[
                     (MujocoRandomLightsWrapper, dict()),
                 ],
             ))
    register(id='SawyerPickupRandomColorsEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=SawyerPickupGoalEnv,
                 env_kwargs=PICKUP_SHARED_KWARGS,
                 wrappers=[
                     (MujocoRandomColorWrapper,
                      dict(
                          color_choices=TABLE_COLORS,
                          geom_names=['tableTop'],
                      )),
                     (MujocoRandomColorWrapper,
                      dict(
                          color_choices=BALL_COLORS,
                          geom_names=['objbox'],
                          site_names=['obj'],
                      )),
                 ],
             ))
    register(id='SawyerPickupRandomLightsColorsEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=SawyerPickupGoalEnv,
                 env_kwargs=PICKUP_SHARED_KWARGS,
                 wrappers=[
                     (MujocoRandomLightsWrapper, dict()),
                     (MujocoRandomColorWrapper,
                      dict(
                          color_choices=TABLE_COLORS,
                          geom_names=['tableTop'],
                      )),
                     (MujocoRandomColorWrapper,
                      dict(
                          color_choices=BALL_COLORS,
                          geom_names=['objbox'],
                          site_names=['obj'],
                      )),
                 ],
             ))

    # Sawyer door environments
    # ========================

    DOOR_SHARED_KWARGS = dict(
        goal_low=(-0.1, 0.45, 0.1, 0),
        goal_high=(0.05, 0.65, .25, .83),
        hand_low=(-0.1, 0.45, 0.1),
        hand_high=(0.05, 0.65, .25),
        max_angle=.83,
        xml_path='sawyer_door_pull_hook.xml',
        reward_type='angle_diff_and_hand_distance',
        reset_free=True,
    )

    register(id='SawyerDoorGoalEnv-v1',
             entry_point='weakly_supervised_control.envs.sawyer_door:SawyerDoorGoalEnv',
             kwargs=DOOR_SHARED_KWARGS)
    register(id='SawyerDoorRandomLightsEnv-v1',
             entry_point=create_wrapped_env,
             kwargs=dict(
                 env_cls=SawyerDoorGoalEnv,
                 env_kwargs=DOOR_SHARED_KWARGS,
                 wrappers=[
                     (MujocoRandomLightsWrapper, dict()),
                 ],
             ))
