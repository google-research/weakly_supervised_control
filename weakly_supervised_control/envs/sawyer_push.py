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

"""Pushing balls with a Sawyer arm.

To run test:
python -m weakly_supervised_control.envs.sawyer_push
"""

import collections
import copy
import os
from typing import List, Tuple

from gym import spaces
import mujoco_py
import numpy as np

from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable

PUCK_TEMPLATE = """
    <body name="puck-{i}" pos="0 0 0.1">
        <joint name="puckjoint-{i}" type="free" limited='false' damping="0" armature="0" />
        <inertial pos="0 0 0" mass=".1" diaginertia="100000 100000 100000"/>
        <geom name="puckbox-{i}"
                type="cylinder"
                pos="0 0 0"
                size="0.04 0.015"
                rgba="{color}"
                contype="2"
                conaffinity="6"
        />
        <site name="puck-{i}" pos="0 0 0" size="0.01" />
    </body>
    <body name="puck-goal-{i}" pos="0 0 0">
        <joint name="puck-goal-joint-{i}" type="free" limited='false'
                damping="0" armature="0" />
        <geom name="puck-goal-marker-{i}" type="sphere" pos="0 0 0" size="0.02"
                rgba=".9 .1 .1 0" contype="0" conaffinity="8"/>
        <site name="puck-goal-site-{i}" pos="0 0 0" size="0.01" rgba="0 0 0 0" />
    </body>
"""

ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets')

XML_TEMPLATE_PATH = os.path.join(ASSETS_PATH, 'sawyer_push_balls_template.xml')

MESHES_PATH = os.path.join(ASSETS_PATH,
                           '../multiworld/envs/assets/meshes/sawyer')

CACHE_PATH_BASE = '/tmp/sawyer_push_generated/push_xml_{}pucks.xml'

COLORS = [
    (0, 0, 1., 1),  # Blue
    (1., 0, 0, 1),  # Red
    (0, 1., 0, 1),  # Green
    (1., 1., 0, 1),  # Yellow
    (1., 0, 1., 1),  # Magenta
    (1., 0.5, 0.5, 1),  # Pink
    (0.5, 0.5, 0.5, 1),  # Gray
    (0.5, 0, 0, 1),  # Brown
    (1., 0.5, 0, 1),  # Orange
]


class SawyerPushGoalBaseEnv(MujocoEnv, Serializable, MultitaskEnv):

    # Mapping of goal puck count to the generated XML.
    _XML_PATH_CACHE = {}

    INIT_HAND_POS = np.array([0, 0.4, 0.02])

    def __init__(
        self,
        puck_count: int = 2,
        reward_info=None,
        frame_skip: int = 50,
        pos_action_scale: float = 2. / 100,
        randomize_goals: bool = True,
        hide_goal: bool = False,
        init_block_low: Tuple[float, float] = (-0.05, 0.55),
        init_block_high: Tuple[float, float] = (0.05, 0.65),
        puck_goal_low: Tuple[float, float] = (-0.05, 0.55),
        puck_goal_high: Tuple[float, float] = (0.05, 0.65),
        hand_goal_low: Tuple[float, float] = (-0.05, 0.55),
        hand_goal_high: Tuple[float, float] = (0.05, 0.65),
        fixed_puck_goal: Tuple[float, float] = (0.05, 0.6),
        fixed_hand_goal: Tuple[float, float] = (-0.05, 0.6),
        mocap_low: Tuple[float, float, float] = (-0.1, 0.5, 0.0),
        mocap_high: Tuple[float, float, float] = (0.1, 0.7, 0.5),
        area_bound_low: Tuple[float, float] = (-0.2, 0.5),
        area_bound_high: Tuple[float, float] = (0.2, 0.7),
        force_puck_in_goal_space=False,
    ):
        self.quick_init(locals())
        self.puck_count = puck_count
        self.reward_info = reward_info
        self.randomize_goals = randomize_goals
        self._pos_action_scale = pos_action_scale
        self.hide_goal = hide_goal

        self.init_block_low = np.array(init_block_low)
        self.init_block_high = np.array(init_block_high)
        self.hand_goal_low = np.array(hand_goal_low)
        self.hand_goal_high = np.array(hand_goal_high)

        self.puck_goal_low = np.array(puck_goal_low * puck_count)
        self.puck_goal_high = np.array(puck_goal_high * puck_count)
        self.fixed_puck_goal = np.array(fixed_puck_goal * puck_count)
        self.fixed_hand_goal = np.array(fixed_hand_goal * puck_count)

        self.mocap_low = np.array(mocap_low)
        self.mocap_high = np.array(mocap_high)
        self.force_puck_in_goal_space = force_puck_in_goal_space

        self._goal_xyxy = self.sample_goal_xyxy()

        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.obs_box = spaces.Box(
            np.array(area_bound_low * (puck_count + 1)),
            np.array(area_bound_high * (puck_count + 1)),
        )
        goal_low = np.concatenate((self.hand_goal_low, self.puck_goal_low))
        goal_high = np.concatenate((self.hand_goal_high, self.puck_goal_high))
        self.goal_box = spaces.Box(goal_low, goal_high)

        self.observation_space = spaces.Dict([
            ('observation', self.obs_box),
            ('state_observation', self.obs_box),
            ('desired_goal', self.goal_box),
            ('state_desired_goal', self.goal_box),
            ('achieved_goal', self.goal_box),
            ('state_achieved_goal', self.goal_box),
        ])

        self.endeff_id = self.model.body_name2id('leftclaw')
        self.hand_goal_id = self.model.body_name2id('hand-goal')
        self.puck_ids = [
            self.model.body_name2id('puck-{}'.format(i))
            for i in range(puck_count)
        ]
        self.puck_goal_ids = [
            self.model.body_name2id('puck-goal-{}'.format(i))
            for i in range(puck_count)
        ]

        self.hand_goal_qpos_index = self.model.get_joint_qpos_addr(
            'hand-goal-joint')[0]
        self.hand_goal_qvel_index = self.model.get_joint_qvel_addr(
            'hand-goal-joint')[0]
        self.puck_qpos_indices = [
            self.model.get_joint_qpos_addr('puckjoint-{}'.format(i))[0]
            for i in range(puck_count)
        ]
        self.puck_qvel_indices = [
            self.model.get_joint_qvel_addr('puckjoint-{}'.format(i))[0]
            for i in range(puck_count)
        ]
        self.puck_goal_qpos_indices = [
            self.model.get_joint_qpos_addr('puck-goal-joint-{}'.format(i))[0]
            for i in range(puck_count)
        ]
        self.puck_goal_qvel_indices = [
            self.model.get_joint_qvel_addr('puck-goal-joint-{}'.format(i))[0]
            for i in range(puck_count)
        ]

        self.reset()
        self.reset_mocap_welds()

    @property
    def factor_names(self):
        return ['hand_x', 'hand_y', 'obj_x', 'obj_y']

    @property
    def model_name(self) -> str:
        """Returns the path to the mujoco xml."""
        if self.puck_count in self._XML_PATH_CACHE:
            return self._XML_PATH_CACHE[self.puck_count]
        generated_path = self._generate_xml(self.puck_count)
        self._XML_PATH_CACHE[self.puck_count] = generated_path
        return generated_path

    @property
    def goal_dim(self) -> int:
        return 2 + 2 * self.puck_count

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()[:2]

    def get_hand_goal_pos(self):
        return self.data.body_xpos[self.hand_goal_id].copy()[:2]

    def get_puck_positions(self) -> List[Tuple[float, float]]:
        return [
            self.data.body_xpos[puck_id].copy()[:2]
            for puck_id in self.puck_ids
        ]

    def get_puck_goal_positions(self) -> List[Tuple[float, float]]:
        return [
            self.data.body_xpos[puck_id].copy()[:2]
            for puck_id in self.puck_goal_ids
        ]

    @property
    def init_angles(self):
        return [
            1.78026069e+00,
            -6.84415781e-01,
            -1.54549231e-01,
            2.30672090e+00,
            1.93111471e+00,
            1.27854012e-01,
            1.49353907e+00,
            1.80196716e-03,
            7.40415706e-01,
            2.09895360e-02,
            9.99999990e-01,
            3.05766105e-05,
            -3.78462492e-06,
            1.38684523e-04,
        ]

    def viewer_setup(self):
        """Sets up the rendering camera."""
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset(self):
        velocities = self.data.qvel.copy()
        positions = self.init_qpos.copy()
        init_arm_pos = self.init_angles
        positions[:14] = init_arm_pos
        expected_nq = len(init_arm_pos) + 14 * self.puck_count
        assert expected_nq == self.model.nq, (expected_nq, self.model.nq)
        self.set_state(positions.flatten(), velocities.flatten())

        self.data.set_mocap_pos('mocap', self.INIT_HAND_POS)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

        # set_state resets the goal xy, so we need to explicit set it again
        self._goal_xyxy = self.sample_goal_for_rollout()
        self.set_goal_xyxy(self._goal_xyxy)
        self.set_puck_xy(self.sample_puck_xy())
        self.reset_mocap_welds()

        return self._get_obs()

    def step(self, a):
        a = np.clip(a, -1, 1)
        mocap_delta_z = 0.06 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((a, np.array([mocap_delta_z])))
        self.mocap_set_action(new_mocap_action[:3] * self._pos_action_scale)
        if self.force_puck_in_goal_space:
            puck_pos = self.get_puck_positions()
            clipped = np.clip(puck_pos, self.puck_goal_low,
                              self.puck_goal_high)
            if not (clipped == puck_pos).all():
                self.set_puck_xy(np.hstack(clipped))
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()

        reward = self.compute_reward(a, obs)
        done = False

        hand_distance = np.linalg.norm(self.get_hand_goal_pos() -
                                       self.get_endeff_pos())
        info = dict(hand_distance=hand_distance)

        puck_positions = self.get_puck_positions()
        puck_goal_positions = self.get_puck_goal_positions()
        for i in range(self.puck_count):
            info['puck{}_distance'.format(i)] = np.linalg.norm(
                puck_goal_positions[i] - puck_positions[i])
            info['touch{}_distance'.format(i)] = np.linalg.norm(
                self.get_endeff_pos() - puck_positions[i])

        return obs, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()[:2]
        b = self.get_puck_positions()
        x = np.concatenate([e] + b)
        g = self._goal_xyxy
        assert x.shape == g.shape, (x.shape, g.shape)

        new_obs = dict(
            observation=x,
            state_observation=x,
            desired_goal=g,
            state_desired_goal=g,
            achieved_goal=x,
            state_achieved_goal=x,
        )
        return new_obs

    def mocap_set_action(self, action):
        pos_delta = action[None]
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_puck_xy(self, pos):
        assert pos.shape == (2 * self.puck_count, ), pos
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        for i in range(self.puck_count):
            i_offset = 2 * i
            puck_pos = pos[i_offset:i_offset + 2]
            i_qp = self.puck_qpos_indices[i]
            i_qv = self.puck_qvel_indices[i]
            qpos[i_qp:i_qp + 3] = np.append(puck_pos, 0.02)
            qpos[i_qp + 3:i_qp + 7] = [1, 0, 0, 0]
            qvel[i_qv:i_qv + 3] = [0, 0, 0]

        self.set_state(qpos, qvel)

    def set_goal_xyxy(self, xyxy):
        assert xyxy.shape == (2 * (1 + self.puck_count), ), xyxy.shape
        self._goal_xyxy = xyxy
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        hand_goal = xyxy[:2]
        i_qp_hg = self.hand_goal_qpos_index
        i_qv_hg = self.hand_goal_qvel_index
        qpos[i_qp_hg:i_qp_hg + 3] = np.append(hand_goal, 0.02)
        qvel[i_qv_hg:i_qv_hg + 3] = np.zeros(3)

        for i in range(self.puck_count):
            i_offset = 2 * (i + 1)
            puck_goal = xyxy[i_offset:i_offset + 2]
            i_qp = self.puck_goal_qpos_indices[i]
            i_qv = self.puck_goal_qvel_indices[i]
            qpos[i_qp:i_qp + 3] = np.append(puck_goal, 0.02)
            qvel[i_qv:i_qv + 3] = np.zeros(3)

        self.set_state(qpos, qvel)

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def compute_rewards(self, action, obs, info=None):
        r = -np.linalg.norm(
            obs['state_achieved_goal'] - obs['state_desired_goal'], axis=1)
        return r

    def compute_reward(self, action, obs, info=None):
        r = -np.linalg.norm(obs['state_achieved_goal'] -
                            obs['state_desired_goal'])
        return r

    """Multitask methods."""

    def sample_goals(self, batch_size):
        """Samples a batch of goals."""
        goals = np.zeros((batch_size, self.goal_box.low.size))
        for b in range(batch_size):
            goals[b, :] = self.sample_goal_xyxy(randomize=True)
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def sample_goal_xyxy(self, randomize: bool = False):
        """Samples a single goal."""
        if self.randomize_goals or randomize:
            hand = np.random.uniform(self.hand_goal_low, self.hand_goal_high)
            puck_goals = np.random.uniform(self.puck_goal_low,
                                           self.puck_goal_high)
            puck_goals = self._separate_puck_positions(puck_goals, random=True)
        else:
            hand = self.fixed_hand_goal.copy()
            puck_goals = self.fixed_puck_goal.copy()
            puck_goals = self._separate_puck_positions(puck_goals)
        return np.hstack([hand, puck_goals])

    def sample_goal_for_rollout(self):
        g = self.sample_goal_xyxy()
        return g

    def sample_puck_xy(self):
        raise NotImplementedError("Not implemented in base class.")

    def get_goal(self):
        return {
            'desired_goal': self._goal_xyxy,
            'state_desired_goal': self._goal_xyxy,
        }

    def set_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.set_goal_xyxy(state_goal)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.set_hand_xy(state_goal[:2])
        self.set_puck_xy(state_goal[2:])

    def convert_obs_to_goals(self, obs):
        return obs

    def set_hand_xy(self, xy):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1], 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()

    def get_diagnostics(self, paths, prefix=""):
        statistics = collections.OrderedDict()
        stats = ['hand_distance']
        for i in range(self.puck_count):
            stats += [
                'puck{}_distance'.format(i),
                'touch{}_distance'.format(i),
            ]
        for stat_name in stats:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(
                create_stats_ordered_dict(
                    '%s%s' % (prefix, stat_name),
                    stat,
                    always_show_all_stats=True,
                ))
            statistics.update(
                create_stats_ordered_dict(
                    'Final %s%s' % (prefix, stat_name),
                    [s[-1] for s in stat],
                    always_show_all_stats=True,
                ))
        return statistics

    def _separate_puck_positions(
            self,
            positions: np.ndarray,
            min_distance: float = 0.08,
            padding: float = 0.02,
            max_iters: int = 50,
            random: bool = False,
    ) -> List[Tuple[float, float]]:
        """Returns the separated goal positions."""
        assert positions.shape == (2 * self.puck_count, ), positions.shape
        puck_positions = []
        for i in range(self.puck_count):
            puck_positions.append(positions[2 * i:2 * i + 2].copy())

        def _get_unclusted(group: List[Tuple[float, float]]):
            assert len(group) >= 1
            if len(group) == 1:
                return group
            n = len(group)
            r = (min_distance + padding) / (2 * np.sin(np.pi / n))
            center = np.mean(group, axis=0)
            assert center.shape == (2, ), center.shape
            offset = 0
            if random:
                offset = np.random.uniform(0, 2 * np.pi)
            results = []
            for i in range(n):
                a = 2 * np.pi * i / n + offset
                xy = np.array([np.cos(a), np.sin(a)])
                results.append(center + r * xy)
            return results

        # Iteratively uncluster puck positions.
        unchecked_pucks = collections.deque(puck_positions)
        all_clusters = []
        iters = 0
        while len(all_clusters) != len(puck_positions) and iters < max_iters:
            for cluster in all_clusters:
                unchecked_pucks.extend(_get_unclusted(cluster))
            all_clusters.clear()
            assert unchecked_pucks

            while unchecked_pucks:
                puck = unchecked_pucks.popleft()
                assert puck.shape == (2, ), puck.shape
                in_cluster = False
                for cluster in all_clusters:
                    for cluster_puck in cluster:
                        distance = np.linalg.norm(cluster_puck - puck)
                        if distance < min_distance:
                            in_cluster = True
                            cluster.append(puck)
                            break
                    if in_cluster:
                        break
                if not in_cluster:
                    all_clusters.append([puck])
            iters += 1
        if iters >= max_iters:
            raise ValueError('Could not uncluster.')

        result = np.hstack([c[0] for c in all_clusters])
        assert result.shape == (2 * self.puck_count, ), result.shape
        return result

    def _generate_xml(self, puck_count: int) -> str:
        """Generates the XML for the given puck count and returns the path"""
        # Generate the XML.
        with open(XML_TEMPLATE_PATH, 'r') as f:
            template = f.read()

        save_path = CACHE_PATH_BASE.format(puck_count)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        assert os.path.isdir(MESHES_PATH), MESHES_PATH

        contents = template.format(
            pucks='\n'.join([
                PUCK_TEMPLATE.format(
                    i=i,
                    color=' '.join(map(str, COLORS[i % len(COLORS)])),
                ) for i in range(self.puck_count)
            ]),
            meshdir=MESHES_PATH,
        )

        print('Saving XML to: {}'.format(save_path))
        with open(save_path, 'w') as f:
            f.write(contents)
        return save_path


class SawyerPushGoalXYEasyEnv(SawyerPushGoalBaseEnv):
    """
    Always start the block in the same position, and use a 40x20 puck space
    """

    def __init__(self, **kwargs):
        self.quick_init(locals())
        default_kwargs = dict(
            puck_goal_low=(-0.2, 0.5),
            puck_goal_high=(0.2, 0.7),
        )
        actual_kwargs = {**default_kwargs, **kwargs}
        SawyerPushGoalBaseEnv.__init__(self, **actual_kwargs)

    def sample_puck_xy(self):
        return self._separate_puck_positions(
            np.array([0, 0.6] * self.puck_count))


class SawyerPushGoalHarderEnv(SawyerPushGoalBaseEnv):
    """
    Fixed initial position, all spaces are 40cm x 20cm
    """

    def __init__(self, **kwargs):
        self.quick_init(locals())
        SawyerPushGoalBaseEnv.__init__(
            self,
            hand_goal_low=(-0.2, 0.5),
            hand_goal_high=(0.2, 0.7),
            puck_goal_low=(-0.2, 0.5),
            puck_goal_high=(0.2, 0.7),
            mocap_low=(-0.2, 0.5, 0.0),
            mocap_high=(0.2, 0.7, 0.5),
            **kwargs,
        )

    def sample_puck_xy(self):
        return self._separate_puck_positions(
            np.array([0, 0.6] * self.puck_count))


if __name__ == '__main__':
    import gym
    from weakly_supervised_control.envs import register_all_envs

    N_PUCKS = 2
    SAMPLE_GOALS = True
    EP_LENGTH = 150
    ENV_ID = 'SawyerPush3PucksRandomLightsEnv-v1'

    register_all_envs()
    env = gym.make(ENV_ID)

    for e in range(20):
        env.reset()

        if SAMPLE_GOALS:
            goal = env.sample_goals(1)
            goal = {k: v[0] for k, v in goal.items()}
            env.set_to_goal(goal)

        for _ in range(EP_LENGTH):
            if not SAMPLE_GOALS:
                env.step(env.action_space.sample())
            env.render()
