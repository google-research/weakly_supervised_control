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
#
#  This file was modified from `https://github.com/google-research/google-research/blob/master/weak_disentangle`.
from weakly_supervised_control.disentanglement.np_data import NpGroundTruthData
from weakly_supervised_control.disentanglement import datasets, networks
import os
import numpy as np
from scipy.stats.stats import pearsonr
import time
from typing import Any, Callable, Dict, List, Optional
import tensorflow as tf
tf.enable_v2_behavior()
tfk = tf.keras


def get_masks(factors="r=0,1,2,3", s_dim=4):
    if "c=" in factors:
        mask_type = "match"
    elif "r=" in factors:
        mask_type = "rank"
    else:
        assert NotImplementedError(factors)

    strategy, factors = factors.split("=")
    masks = np.array(list(map(int, factors.split(","))))
    print("masks:", masks)
    return masks, mask_type


class DisentanglementModel():
    def __init__(
            self,
            x_shape: List[int],
            num_factors: int,
            factors: str = None,
            n_dim: int = 0,  # nuisance parameter
            enc_lr: float = 1e-3,
            dis_lr: float = 1e-3,
            gen_lr: float = 1e-3,
            enc_width: int = 1,
            dis_width: int = 2):
        self.n_dim = n_dim
        self.s_dim = num_factors
        self.x_shape = x_shape

        if factors is None:
            factors = 'r=' + ','.join(map(str, range(num_factors)))
        self.masks, self.mask_type = get_masks(factors, self.s_dim)
        self.y_dim = len(self.masks)

        # Initialize networks.
        if len(self.x_shape) == 3:
            self.dis = networks.Discriminator(
                self.x_shape, self.y_dim, mask_type=self.mask_type, width=dis_width)
            self.gen = networks.Generator(
                self.x_shape, self.s_dim + self.n_dim)
            # Encoder ignores nuisance param
            self.enc = networks.Encoder(
                self.x_shape, self.s_dim, width=enc_width)
        else:
            assert len(self.x_shape) == 4
            self.dis = networks.SiameseDiscriminator(
                self.x_shape, self.y_dim, mask_type=self.mask_type, width=dis_width)
            self.gen = networks.SiameseGenerator(
                self.x_shape, self.s_dim + self.n_dim)
            # Encoder ignores nuisance param
            self.enc = networks.SiameseEncoder(
                self.x_shape, self.s_dim, width=enc_width)

        print(self.dis.read(self.dis.WITH_VARS))
        print(self.gen.read(self.gen.WITH_VARS))
        print(self.enc.read(self.enc.WITH_VARS))

        # Initialize optimizers.
        self.gen_opt = tfk.optimizers.Adam(
            learning_rate=gen_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
        self.dis_opt = tfk.optimizers.Adam(
            learning_rate=dis_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
        self.enc_opt = tfk.optimizers.Adam(
            learning_rate=enc_lr, beta_1=0.5, beta_2=0.999, epsilon=1e-8)

        self.ckpt_root = tf.train.Checkpoint(dis=self.dis, dis_opt=self.dis_opt,
                                             gen=self.gen, gen_opt=self.gen_opt,
                                             enc=self.enc, enc_opt=self.enc_opt)

    def load_checkpoint(self, ckpt_dir):
        # Load from checkpoint.
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt is None:
            print("Starting a completely new model")
            return 1
        else:
            print("Restarting from {}".format(latest_ckpt))
            self.ckpt_root.restore(latest_ckpt)
            return self.ckpt_root.save_counter

    def save_checkpoint(self, ckpt_prefix):
        print("Saved to", self.ckpt_root.save(ckpt_prefix))

    @tf.function
    def gen_eval(self, z):
        self.gen.eval()
        return self.gen(z)

    @tf.function
    def enc_eval(self, x):
        self.enc.eval()
        return self.enc(x).mean()

    def evaluate_correlation(self, dset: NpGroundTruthData, sample_size: int = 1000):
        # Returns Pearson-r correlation between factors and latents.
        n = min(sample_size, dset.size)
        idx = np.random.choice(range(dset.size), n, replace=False)
        observations = dset.data[idx]
        factors = dset.factors[idx]
        factor_keys = list(range(dset.num_factors))

        z = self.enc_eval(tf.convert_to_tensor(observations)).numpy()

        evals = {}
        for i in range(z.shape[1]):
            eval_key = 'pearsonr_correlation_{}_{}'.format(i, factor_keys[i])
            corr, _ = pearsonr(z[:, i], factors[:, i])
            evals[eval_key] = corr
        return evals

    def get_latent_range(self, x: np.ndarray):
        z_all = self.enc_eval(tf.convert_to_tensor(x)).numpy()
        z_min = np.min(z_all, axis=0)
        z_max = np.max(z_all, axis=0)
        return z_min, z_max


class DisentanglementTrainer():
    def __init__(self, model, dset,
                 sample_factors_kwargs={},
                 batch_size: int = 64):
        self.model = model
        self.dset = dset
        self.masks = self.model.masks
        self.mask_type = self.model.mask_type
        self.s_dim = self.model.s_dim
        self.z_dim = self.model.s_dim + self.model.n_dim
        self.batch_size = batch_size

        self.batches = iter(datasets.paired_data_generator(
            self.dset, self.masks,
            mask_type=self.mask_type,
            **sample_factors_kwargs).repeat().batch(self.batch_size).prefetch(1000))

        self.enc = self.model.enc
        self.gen = self.model.gen
        self.dis = self.model.dis
        self.enc_opt = self.model.enc_opt
        self.gen_opt = self.model.gen_opt
        self.dis_opt = self.model.dis_opt

    @tf.function
    def train_gen_step(self, x1_real, x2_real, y_real):
        self.gen.train()
        self.dis.train()
        self.enc.train()

        targets_real = tf.ones((self.batch_size, 1))
        targets_fake = tf.zeros((self.batch_size, 1))
        targets = tf.concat((targets_real, targets_fake), axis=0)

        # Alternate discriminator step and generator step
        with tf.GradientTape(persistent=True) as tape:
            # Generate
            z1, z2, y_fake = datasets.paired_randn(
                self.batch_size, self.z_dim, self.masks, mask_type=self.mask_type)
            x1_fake = tf.stop_gradient(self.gen(z1))
            x2_fake = tf.stop_gradient(self.gen(z2))

            # Discriminate
            x1 = tf.concat((x1_real, x1_fake), 0)
            x2 = tf.concat((x2_real, x2_fake), 0)
            y = tf.concat((y_real, y_fake), 0)
            logits = self.dis(x1, x2, y)

            # Encode
            p_z = self.enc(x1_fake)

            dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=targets))
            # Encoder ignores nuisance parameters (if they exist)
            enc_loss = -tf.reduce_mean(p_z.log_prob(z1[:, :self.s_dim]))

            dis_grads = tape.gradient(dis_loss, self.dis.trainable_variables)
            enc_grads = tape.gradient(enc_loss, self.enc.trainable_variables)

            self.dis_opt.apply_gradients(
                zip(dis_grads, self.dis.trainable_variables))
            self.enc_opt.apply_gradients(
                zip(enc_grads, self.enc.trainable_variables))

        with tf.GradientTape(persistent=False) as tape:
            # Generate
            z1, z2, y_fake = datasets.paired_randn(
                self.batch_size, self.z_dim, self.masks, mask_type=self.mask_type)
            x1_fake = self.gen(z1)
            x2_fake = self.gen(z2)

            # Discriminate
            logits_fake = self.dis(x1_fake, x2_fake, y_fake)

            gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_fake, labels=targets_real))

        gen_grads = tape.gradient(gen_loss, self.gen.trainable_variables)
        self.gen_opt.apply_gradients(
            zip(gen_grads, self.gen.trainable_variables))

        return dict(gen_loss=gen_loss, dis_loss=dis_loss, enc_loss=enc_loss)

    def train_batch(self):
        stopwatch = time.time()
        x1, x2, y = next(self.batches)
        losses = self.train_gen_step(x1, x2, y)
        train_time = time.time() - stopwatch
        return losses, train_time
