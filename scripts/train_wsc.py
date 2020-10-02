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
python -m scripts.train_wsc --config configs/Push1.txt
python -m scripts.train_wsc --config configs/Hardware.txt --render
"""

import os

import click
import gym
import numpy as np
import torch

from tensorflow.io import gfile

from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger, create_exp_name
# from rlkit.samplers.data_collector.vae_env import VAEWrappedEnvPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import FlattenMlp
import rlkit.torch.pytorch_util as ptu
# from weakly_supervised_control.policy import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
# from rlkit.torch.skewfit.online_vae_algorithm import OnlineVaeAlgorithm
from weakly_supervised_control.vae.online_vae_algorithm import OnlineVaeAlgorithm
from weakly_supervised_control.vae.path_collector import VAEWrappedEnvPathCollector

from weakly_supervised_control.envs import register_all_envs
from weakly_supervised_control.vae.conv_vae import VAE
from weakly_supervised_control.vae.online_vae_replay_buffer import ReplayBuffer
from weakly_supervised_control.vae.vae_trainer import VAETrainer
from weakly_supervised_control.envs.disentangled_env import DisentangledEnv
from weakly_supervised_control.experiment_utils import (
    create_disentangled_env,
    disable_tensorflow_gpu,
    load_config,
    load_dset,
    load_disentanglement_model,
    load_experiment,
    train_vae,
)


@click.command()
@click.option('--config', '-c', type=str)
@click.option('--output',
              '-o',
              type=str,
              default='/tmp/weakly_supervised_control/train_wsc',
              help="Output directory")
@click.option('--render', is_flag=True, default=False, type=bool)
def main(config: str, output: str, render: bool):
    disable_tensorflow_gpu()

    variant = load_config(config)
    disentanglement_indices = variant['her_variant'].get(
        'disentanglement_indices', None)
    desired_goal_key = variant['her_variant']['desired_goal_key']
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    # Load datasets.
    train_dset = load_dset(
        variant['env']['train_dataset'],
        stack_images=variant['env'].get('stack_images', False))
    eval_dset = load_dset(
        variant['env']['eval_dataset'],
        stack_images=variant['env'].get('stack_images', False))

    # Set logging.
    log_dir = create_exp_name(os.path.join(output, variant['env']['env_id']))
    print('Logging to:', log_dir)
    setup_logger(log_dir=log_dir, variant=variant,
                 snapshot_mode='last')
    ptu.set_gpu_mode(torch.cuda.is_available())

    # Load disentanglement model and get disentanglement latent space.
    disentanglement_model = None
    disentanglement_space = None
    internal_keys = None
    goal_keys = None
    if variant['her_variant']['reward_params']['type'] in ['disentangled_distance', 'latent_and_disentangled_distance']:
        assert disentanglement_indices is not None
        disentanglement_model, disentanglement_space = load_disentanglement_model(
            train_dset,
            model_kwargs=variant['disentanglement'].get('model_kwargs', {}),
            factor_indices=disentanglement_indices,
            model_path=variant['env']['disentanglement_model_path'],
        )
        internal_keys = ['disentangled_achieved_goal',
                         'disentangled_desired_goal']
        goal_keys = ['disentangled_desired_goal']

    experiment_dir = variant.get('saved_experiment_dir', None)
    if experiment_dir is not None:
        # Read saved models from file.
        _, params, vae = load_experiment(experiment_dir)
        vae = params['vae']
        qf1 = params['trainer/qf1']
        qf2 = params['trainer/qf2']
        target_qf1 = params['trainer/target_qf1']
        target_qf2 = params['trainer/target_qf2']
        policy = params['trainer/policy']
        replay_buffer_snapshot = params['replay_buffer']

        # Create VAE trainer. (Note: This does not train the VAE.)
        vae, vae_trainer = train_vae(
            vae,
            dset=train_dset,
            vae_trainer_kwargs=variant['vae']['vae_trainer_kwargs'])

        # Recreate the DisentangledEnv, since saved_env has incorrect camera view for some reason
        env = create_disentangled_env(
            params['exploration/env'].wrapped_env.wrapped_env,
            vae,
            eval_dset=eval_dset,
            disentangled_env_kwargs=dict(
                disentanglement_model=disentanglement_model,
                disentanglement_space=disentanglement_space,
                disentanglement_indices=disentanglement_indices,
                desired_goal_key=desired_goal_key,
                reward_params=variant['her_variant']['reward_params'],
                **variant['env']['vae_wrapped_env_kwargs']
            ))
    else:
        # Create and pre-train VAE.
        vae = VAE(
            x_shape=train_dset.observation_shape,
            representation_size=variant['her_variant']['representation_size'],
            num_factors=train_dset.num_factors,
            imsize=variant['env']['imsize'],
            **variant['vae']['vae_kwargs'])
        vae.to(ptu.device)
        vae, vae_trainer = train_vae(
            vae,
            dset=train_dset,
            **variant['vae']['pretrain_kwargs'],
            vae_trainer_kwargs=variant['vae']['vae_trainer_kwargs'])

        # Create the environment.
        register_all_envs()
        wrapped_env = gym.make(variant['env']['env_id'])
        env = create_disentangled_env(
            wrapped_env,
            vae,
            eval_dset=eval_dset,
            disentangled_env_kwargs=dict(
                disentanglement_model=disentanglement_model,
                disentanglement_space=disentanglement_space,
                disentanglement_indices=disentanglement_indices,
                desired_goal_key=desired_goal_key,
                reward_params=variant['her_variant']['reward_params'],
                render_rollouts=render,
                render_goals=render,
                **variant['env']['vae_wrapped_env_kwargs']
            ))

        # Initialize policy model.
        hidden_sizes = variant.get('hidden_sizes', [400, 300])
        latent_dim = env.observation_space.spaces['latent_observation'].low.size
        goal_dim = env.observation_space.spaces[desired_goal_key].low.size
        action_dim = env.action_space.low.size

        mlp_kwargs = dict(
            input_size=latent_dim + goal_dim + action_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
        )
        qf1 = FlattenMlp(**mlp_kwargs)
        qf2 = FlattenMlp(**mlp_kwargs)
        target_qf1 = FlattenMlp(**mlp_kwargs)
        target_qf2 = FlattenMlp(**mlp_kwargs)

        policy_kwargs = dict(
            obs_dim=latent_dim + goal_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
        policy = TanhGaussianPolicy(**policy_kwargs)

        replay_buffer_snapshot = None

    replay_buffer = ReplayBuffer(
        vae=vae,
        env=env,
        snapshot=replay_buffer_snapshot,
        observation_key='latent_observation',
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        internal_keys=internal_keys,
        goal_keys=goal_keys,
        fraction_goals_env_goals=variant['her_variant']['fraction_goals_env_goals'],
        fraction_goals_rollout_goals=variant['her_variant']['fraction_goals_rollout_goals'],
        relabeling_goal_sampling_mode='custom_goal_sampler',
        power=variant['her_variant']['power'],
        **variant['replay_buffer_kwargs'])
    max_path_length = variant['env']['max_path_length']

    trainer = SACTrainer(env=env,
                         policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         target_qf1=target_qf1,
                         target_qf2=target_qf2,
                         **variant['twin_sac_trainer_kwargs'])
    trainer = HERTrainer(trainer)
    eval_path_collector = VAEWrappedEnvPathCollector(
        'presampled',  # Use presampled images from eval_dataset
        env,
        MakeDeterministic(policy),
        max_path_length,
        observation_key='latent_observation',
        desired_goal_key=desired_goal_key,
    )

    exploration_goal_sampler = variant['her_variant']['exploration_goal_sampler']
    if exploration_goal_sampler == 'replay_buffer':
        expl_goal_sampling_mode = 'custom_goal_sampler'
        env.custom_goal_sampler = replay_buffer.sample_buffer_goals
    else:
        assert exploration_goal_sampler in ['vae_prior', 'observation_space']
        expl_goal_sampling_mode = exploration_goal_sampler

    expl_path_collector = VAEWrappedEnvPathCollector(
        expl_goal_sampling_mode,
        env,
        policy,
        max_path_length,
        observation_key='latent_observation',
        desired_goal_key=desired_goal_key,
    )

    algorithm = OnlineVaeAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        vae=vae,
        vae_trainer=vae_trainer,
        max_path_length=max_path_length,
        **variant['algo_kwargs'])

    algorithm.to(ptu.device)
    vae.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    main()
