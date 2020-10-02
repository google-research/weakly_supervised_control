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
python -m scripts.train_disentanglement -c configs/Push1.txt
python -m scripts.train_disentanglement -c configs/Hardware.txt
ace run-cloud train_disentanglement -c configs/PickupLightsColors.txt
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
from absl import app
import tensorflow.io.gfile as gfile
from tqdm import tqdm

from weakly_supervised_control.experiment_utils import load_config, load_dset, load_disentanglement_model
from weakly_supervised_control.disentanglement.model import DisentanglementTrainer
tf.enable_v2_behavior()
tfk = tf.keras


@click.command()
@click.option('--config', '-c', type=str)
@click.option('--output',
              '-o',
              type=str,
              default='/tmp/weakly_supervised_control/disentangled_model',
              help="Output directory")
def main(config: str,
         output: str):
    variant = load_config(config)
    print(variant)
    train_data_path = variant['env']['train_dataset']
    eval_data_path = variant['env'].get('eval_dataset', train_data_path)
    factors = variant['env']['disentanglement_factors']
    batch_size = variant['disentanglement'].get('batch_size', 64)
    iterations = variant['disentanglement'].get('iterations', 100)
    iter_log = variant['disentanglement'].get('iter_log', 1000)
    iter_save = variant['disentanglement'].get('iter_save', int(1e4))

    # Create output directories.
    output += '/' + '{}-{}-{}'.format(
        os.path.basename(train_data_path).split('.')[0],
        factors,
        time.strftime("%Y%m%d-%H%M%S"))
    ckptdir = os.path.join(output, "ckptdir")
    vizdir = os.path.join(output, "vizdir")
    ckpt_prefix = os.path.join(ckptdir, "model")
    gfile.makedirs(ckptdir)
    gfile.makedirs(vizdir)

    # Initialize dataset, model, and trainer.
    if eval_data_path is None:
        eval_data_path = train_data_path
    train_dset = load_dset(
        train_data_path, stack_images=variant['env'].get('stack_images', False))
    eval_dset = load_dset(
        eval_data_path, stack_images=variant['env'].get('stack_images', False))
    model, _ = load_disentanglement_model(
        train_dset, factors=factors, model_kwargs=variant['disentanglement'].get('model_kwargs', {}))
    trainer = DisentanglementTrainer(model, train_dset, batch_size=batch_size)

    summary_writer = tf.contrib.summary.create_file_writer(
        os.path.join(output, 'tb'), max_queue=1)

    def write_summary(d: dict, step: int, scalar_prefix: str = ""):
        with summary_writer.as_default():
            for k, v in d.items():
                tf.compat.v2.summary.scalar(scalar_prefix + k, v, step=step)
        summary_writer.flush()

    # Load model from checkpoint.
    save_counter = model.load_checkpoint(ckptdir)
    train_range = range(iter_save * int(save_counter - 1), iterations)

    total_train_time = 0
    start_time = time.time()
    for global_step in train_range:
        losses, train_time = trainer.train_batch()
        total_train_time += train_time

        if (global_step + 1) % iter_log == 0 or global_step == 0:
            write_summary(losses, global_step, 'train/')

            elapsed_time = time.time() - start_time
            write_summary({
                'elapsed_time': elapsed_time,
                'elapsed_time_iter_s': global_step / elapsed_time,
                'train_time_iter_s': global_step / total_train_time,
            }, global_step, 'time/')

            train_corr = model.evaluate_correlation(train_dset)
            eval_corr = model.evaluate_correlation(eval_dset)
            write_summary(train_corr, global_step, "train_corr/")
            write_summary(eval_corr, global_step, "eval_corr/")

            train_corr_str = ', '.join(
                [f'{corr:.3f}' for corr in train_corr.values()])
            eval_corr_str = ', '.join(
                [f'{corr:.3f}' for corr in eval_corr.values()])
            print(
                f"Iter: {global_step:06d}, Elapsed: {elapsed_time:.2e}, Total: {total_train_time:.2e}, Gen: {losses['gen_loss']:.3f}, Dis: {losses['dis_loss']:.3f}, Enc: {losses['enc_loss']:.3f}, Corr: {train_corr_str} (Train), {eval_corr_str} (Eval)")

        if (global_step + 1) % iter_save == 0 or global_step == 0:
            model.save_checkpoint(ckpt_prefix)


if __name__ == "__main__":
    main()
