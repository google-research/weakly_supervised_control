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
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.samplers.data_collector import GoalConditionedPathCollector


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            goal_sampling_mode,
            env: VAEWrappedEnv,
            policy,
            decode_goals=False,
            **kwargs
    ):
        super().__init__(env, policy, **kwargs)
        self._goal_sampling_mode = goal_sampling_mode
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
            # epoch_paths=self._epoch_paths,
            # num_steps_total=self._num_steps_total,
            # num_paths_total=self._num_paths_total,
        )
