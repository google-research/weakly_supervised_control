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
"""Models."""

import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from collections import OrderedDict
import numpy as np

import weakly_supervised_control.disentanglement.tensorsketch as ts

tfd = tfp.distributions


def reset_parameters(m):
    m.reset_parameters()


class Encoder(ts.Module):
    def __init__(self, x_shape, z_dim, width=1, spectral_norm=True):
        super().__init__()
        self.net = ts.Sequential(
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Flatten(),
            ts.Dense(128 * width), ts.LeakyReLU(),
            ts.Dense(2 * z_dim)
        )

        if spectral_norm:
            self.net.apply(ts.SpectralNorm.add, targets=ts.Affine)

        x_shape = [1] + [int(x) for x in x_shape]
        self.build(x_shape)
        self.apply(reset_parameters)

    def forward(self, x):
        x = tf.cast(x, tf.float32)
        h = self.net(x)
        a, b = tf.split(h, 2, axis=-1)
        return tfd.MultivariateNormalDiag(
            loc=a,
            scale_diag=tf.nn.softplus(b) + 1e-8)


class Discriminator(ts.Module):
    def __init__(self, x_shape, y_dim, width=2, share_dense=True,
                 uncond_bias=False, cond_bias=False, mask_type="match"):
        super().__init__()
        self.y_dim = y_dim
        self.mask_type = mask_type
        self.body = ts.Sequential(
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Flatten(),
        )

        if share_dense:
            self.body.append(ts.Dense(128 * width), ts.LeakyReLU())

        if mask_type == "match":
            self.neck = ts.Sequential(
                ts.Dense(128 * width), ts.LeakyReLU(),
                ts.Dense(128 * width), ts.LeakyReLU(),
            )

            self.head_uncond = ts.Dense(1, bias=uncond_bias)
            self.head_cond = ts.Dense(128 * width, bias=cond_bias)

            for m in (self.body, self.neck, self.head_uncond):
                m.apply(ts.SpectralNorm.add, targets=ts.Affine)
            ts.WeightNorm.add(self.head_cond)
            x_shape = [1] + [int(x) for x in x_shape]
            y_shape = (1, y_dim)
            # x_shape, y_shape = [[1] + [x] for x in x_shape][0], ((1,), tf.int32)

        elif mask_type == "rank":
            self.body.append(
                ts.Dense(128 * width), ts.LeakyReLU(),
                ts.Dense(128 * width), ts.LeakyReLU(),
                ts.Dense(1 + y_dim, bias=uncond_bias)
            )

            self.body.apply(ts.SpectralNorm.add, targets=ts.Affine)
            x_shape = [1] + [int(x) for x in x_shape]
            y_shape = (1, y_dim)
        self.build(x_shape, x_shape, y_shape)
        self.apply(reset_parameters)

    def forward(self, x1, x2, y):
        if self.mask_type == "match":
            h = self.body(tf.concat((x1, x2), axis=0))
            h1, h2 = tf.split(h, 2, axis=0)
            h = self.neck(tf.concat((h1, h2), axis=-1))
            o_uncond = self.head_uncond(h)

            # Hacky solution
            try:
                # Eager mode
                w = self.head_cond(tf.one_hot(
                    tf.cast(y, tf.float32).numpy(),
                    tf.cast(self.y_dim, tf.float32).numpy()))
            except:
                # Graph mode
                w = self.head_cond(tf.one_hot(y, self.y_dim))
            o_cond = tf.reduce_sum(h * w, axis=-1, keepdims=True)
            return o_uncond + o_cond

        elif self.mask_type == "rank":
            h = self.body(tf.concat((x1, x2), axis=0))
            h1, h2 = tf.split(h, 2, axis=0)
            o1, z1 = tf.split(h1, (1, self.y_dim), axis=-1)
            o2, z2 = tf.split(h2, (1, self.y_dim), axis=-1)
            y_pm = y * 2 - 1  # convert from {0, 1} to {-1, 1}
            y_pm = tf.cast(y_pm, tf.float32)
            if len(y_pm.shape) == 1:
                y_pm = tf.expand_dims(y_pm, 1)
            diff = (z1 - z2) * y_pm
            o_diff = tf.reduce_sum(diff, axis=-1, keepdims=True)
            return o1 + o2 + o_diff

    def expose_encoder(self, x):
        h = self.body(x)
        _, z = tf.split(h, (1, self.y_dim), axis=-1)
        return z


class Generator(ts.Module):
    def __init__(self, x_shape, z_dim, batch_norm=True):
        super().__init__()
        ch = x_shape[-1]
        if x_shape[0] == 64:
            self.net = ts.Sequential(
                ts.Dense(128), ts.ReLU(),
                ts.Dense(4 * 4 * 64), ts.ReLU(), ts.Reshape((-1, 4, 4, 64)),
                ts.ConvTranspose2d(64, 4, 2, "same"), ts.LeakyReLU(),
                ts.ConvTranspose2d(32, 4, 2, "same"), ts.LeakyReLU(),
                ts.ConvTranspose2d(32, 4, 2, "same"), ts.LeakyReLU(),
                ts.ConvTranspose2d(ch, 4, 2, "same"), ts.Sigmoid(),
            )
        elif x_shape[1] == 48:
            self.net = ts.Sequential(
                ts.Dense(128), ts.ReLU(),
                ts.Dense(3 * 3 * 64), ts.ReLU(), ts.Reshape((-1, 3, 3, 64)),
                ts.ConvTranspose2d(32, 3, 2, "same"), ts.LeakyReLU(),
                ts.ConvTranspose2d(16, 3, 2, "same"), ts.LeakyReLU(),
                ts.ConvTranspose2d(ch, 6, 4, "same"), ts.Sigmoid(),
            )
        else:
            raise NotImplementedError(x_shape)

        # Add batchnorm post-activation (attach to activation out_hook)
        if batch_norm:
            self.net.apply(ts.BatchNorm.add, targets=(ts.ReLU, ts.LeakyReLU))

        self.build((1, z_dim))
        self.apply(reset_parameters)

    def forward(self, z):
        return self.net(z)


class SiameseEncoder(ts.Module):
    def __init__(self, x_shape, z_dim, width=1, spectral_norm=True):
        assert len(x_shape) == 4
        assert x_shape[0] == 2

        super().__init__()
        self.conv1 = ts.Sequential(
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Flatten(),
        )
        self.conv2 = ts.Sequential(
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Flatten(),
        )
        self.dense = ts.Sequential(
            ts.Dense(128 * width), ts.LeakyReLU(),
            ts.Dense(2 * z_dim)
        )

        if spectral_norm:
            self.conv1.apply(ts.SpectralNorm.add, targets=ts.Affine)
            self.conv2.apply(ts.SpectralNorm.add, targets=ts.Affine)
            self.dense.apply(ts.SpectralNorm.add, targets=ts.Affine)

        x_shape = [1] + [int(x) for x in x_shape]
        self.build(x_shape)
        self.apply(reset_parameters)

    def forward(self, x):
        x = tf.cast(x, tf.float32)

        # x1, x2 are of shape (B, C, 48, 48, 3).
        # Split into C tensors of shape (B, 48, 48, 3).
        ch = x.shape[1]
        x = [tf.squeeze(_x, axis=1) for _x in tf.split(x, ch, axis=1)]

        h1 = self.conv1(x[0])
        h2 = self.conv2(x[1])
        h = self.dense(h1 + h2)

        a, b = tf.split(h, 2, axis=-1)
        return tfd.MultivariateNormalDiag(
            loc=a,
            scale_diag=tf.nn.softplus(b) + 1e-8)


class SiameseDiscriminator(ts.Module):
    def __init__(self, x_shape, y_dim, width=2, share_dense=True,
                 uncond_bias=False, cond_bias=False, mask_type="match"):
        super().__init__()
        assert len(x_shape) == 4
        assert x_shape[0] == 2

        self.y_dim = y_dim
        self.mask_type = mask_type
        self.conv1 = ts.Sequential(
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Flatten(),
        )
        self.conv2 = ts.Sequential(
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(32 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Conv2d(64 * width, 4, 2, "same"), ts.LeakyReLU(),
            ts.Flatten(),
        )

        if share_dense:
            self.conv1.append(ts.Dense(128 * width), ts.LeakyReLU())
            self.conv2.append(ts.Dense(128 * width), ts.LeakyReLU())

        self.dense = ts.Sequential(
            ts.Dense(128 * width), ts.LeakyReLU(),
            ts.Dense(128 * width), ts.LeakyReLU(),
            ts.Dense(1 + y_dim, bias=uncond_bias)
        )

        self.conv1.apply(ts.SpectralNorm.add, targets=ts.Affine)
        self.conv2.apply(ts.SpectralNorm.add, targets=ts.Affine)
        self.dense.apply(ts.SpectralNorm.add, targets=ts.Affine)

        x_shape = [1] + [int(x) for x in x_shape]
        y_shape = (1, y_dim)
        self.build(x_shape, x_shape, y_shape)
        self.apply(reset_parameters)

    def forward(self, x1, x2, y):
        # x1, x2 are of shape (B, C, 48, 48, 3).
        # Split into C tensors of shape (B, 48, 48, 3).
        ch = x1.shape[1]
        assert ch == x2.shape[1]
        x1 = [tf.squeeze(_x, axis=1) for _x in tf.split(x1, ch, axis=1)]
        x2 = [tf.squeeze(_x, axis=1) for _x in tf.split(x2, ch, axis=1)]

        c1 = self.conv1(tf.concat((x1[0], x2[0]), axis=0))
        c2 = self.conv2(tf.concat((x1[1], x2[1]), axis=0))
        h = self.dense(c1 + c2)

        h1, h2 = tf.split(h, 2, axis=0)
        o1, z1 = tf.split(h1, (1, self.y_dim), axis=-1)
        o2, z2 = tf.split(h2, (1, self.y_dim), axis=-1)
        y_pm = y * 2 - 1  # convert from {0, 1} to {-1, 1}
        y_pm = tf.cast(y_pm, tf.float32)
        if len(y_pm.shape) == 1:
            y_pm = tf.expand_dims(y_pm, 1)
        diff = (z1 - z2) * y_pm
        o_diff = tf.reduce_sum(diff, axis=-1, keepdims=True)
        return o1 + o2 + o_diff

    def expose_encoder(self, x):
        # x is of shape (B, C, 48, 48, 3).
        # Split into C tensors of shape (B, 48, 48, 3).
        ch = x.shape[1]
        x = [tf.squeeze(_x, axis=1) for _x in tf.split(x, ch, axis=1)]

        c1 = self.conv1(x[0])
        c2 = self.conv1(x[1])
        h = self.dense(c1 + c2)

        _, z = tf.split(h, (1, self.y_dim), axis=-1)
        return z


class SiameseGenerator(ts.Module):
    def __init__(self, x_shape, z_dim, batch_norm=True):
        assert len(x_shape) == 4
        assert x_shape[0] == 2

        super().__init__()
        ch = x_shape[-1]
        self.dense = ts.Sequential(
            ts.Dense(128), ts.ReLU(),
            ts.Dense(3 * 3 * 64), ts.ReLU(), ts.Reshape((-1, 3, 3, 64)),
        )
        self.deconv1 = ts.Sequential(
            ts.ConvTranspose2d(32, 3, 2, "same"), ts.LeakyReLU(),
            ts.ConvTranspose2d(16, 3, 2, "same"), ts.LeakyReLU(),
            ts.ConvTranspose2d(ch, 6, 4, "same"), ts.Sigmoid(),
        )
        self.deconv2 = ts.Sequential(
            ts.ConvTranspose2d(32, 3, 2, "same"), ts.LeakyReLU(),
            ts.ConvTranspose2d(16, 3, 2, "same"), ts.LeakyReLU(),
            ts.ConvTranspose2d(ch, 6, 4, "same"), ts.Sigmoid(),
        )

        # Add batchnorm post-activation (attach to activation out_hook)
        if batch_norm:
            self.dense.apply(ts.BatchNorm.add, targets=(ts.ReLU, ts.LeakyReLU))
            self.deconv1.apply(
                ts.BatchNorm.add, targets=(ts.ReLU, ts.LeakyReLU))
            self.deconv2.apply(
                ts.BatchNorm.add, targets=(ts.ReLU, ts.LeakyReLU))

        self.build((1, z_dim))
        self.apply(reset_parameters)

    def forward(self, z):
        h = self.dense(z)
        x1 = self.deconv1(h)  # (B, 48, 48, 3)
        x2 = self.deconv2(h)  # (B, 48, 48, 3)

        # Concatenate channels
        x1 = tf.expand_dims(x1, axis=1)
        x2 = tf.expand_dims(x2, axis=1)
        x = tf.concat([x1, x2], axis=1)
        return x
