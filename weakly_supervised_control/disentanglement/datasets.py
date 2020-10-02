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
"""Dataset-specific utilities."""

import numpy as np
import tensorflow as tf


def sample_match_factors(dset, batch_size, masks, random_state, **sample_factors_kwargs):
    factor1 = dset.sample_factors(
        batch_size, random_state, **sample_factors_kwargs)
    factor2 = dset.sample_factors(
        batch_size, random_state, **sample_factors_kwargs)
    mask_idx = np.random.choice(len(masks), batch_size)
    mask = masks[mask_idx]
    factor2 = factor2 * mask + factor1 * (1 - mask)
    factors = np.concatenate((factor1, factor2), 0)
    return factors, mask_idx


def sample_rank_factors(dset, batch_size, masks, random_state, **sample_factors_kwargs):
    # We assume for ranking that masks is just a list of indices
    factors = dset.sample_factors(
        2 * batch_size, random_state, **sample_factors_kwargs)
    factor1, factor2 = np.split(factors, 2)
    y = (factor1 > factor2)[:, masks].astype(np.float32)
    return factors, y


def sample_match_images(dset, batch_size, masks, random_state, **sample_factors_kwargs):
    return dset.sample_match_pair(1, random_state)


def sample_rank_images(dset, batch_size, masks, random_state, **sample_factors_kwargs):
    x1, x2, y = dset.sample_rank_pair(1, masks, random_state)
    return x1, x2, y


def sample_images(dset, batch_size, random_state, **sample_factors_kwargs):
    return dset.sample(batch_size, random_state)


def paired_data_generator(dset, masks, random_seed=None, mask_type="match", **sample_factors_kwargs):
    if mask_type == "match":
        return match_data_generator(dset, masks, random_seed, **sample_factors_kwargs)
    elif mask_type == "rank":
        return rank_data_generator(dset, masks, random_seed, **sample_factors_kwargs)
    elif mask_type == "label":
        return label_data_generator(dset, masks, random_seed, **sample_factors_kwargs)


def match_data_generator(dset, masks, random_seed=None, **sample_factors_kwargs):
    def generator():
        random_state = np.random.RandomState(random_seed)

        while True:
            # Returning x1[0] and x2[0] removes batch dimension
            x1, x2, idx = sample_match_images(
                dset, 1, masks, random_state, **sample_factors_kwargs)
            yield x1[0], x2[0], idx.item(0)

    return tf.data.Dataset.from_generator(
        generator,
        (tf.float32, tf.float32, tf.int32),
        output_shapes=(dset.observation_shape, dset.observation_shape, ()))


def rank_data_generator(dset, masks, random_seed=None, **sample_factors_kwargs):
    def generator():
        random_state = np.random.RandomState(random_seed)

        while True:
            # Note: remove batch dimension by returning x1[0], x2[0], y[0]
            x1, x2, y = sample_rank_images(
                dset, 1, masks, random_state, **sample_factors_kwargs)
            yield x1[0], x2[0], y[0]

    y_dim = len(masks)  # Remember, masks is just a list
    return tf.data.Dataset.from_generator(
        generator,
        (tf.float32, tf.float32, tf.float32),
        output_shapes=(dset.observation_shape, dset.observation_shape, (y_dim,)))


def label_data_generator(dset, masks, random_seed=None, **sample_factors_kwargs):
    # Normalize the factors using mean and stddev
    m, s = [], []
    for factor_size in dset.factors_num_values:
        factor_values = list(range(factor_size))
        m.append(np.mean(factor_values))
        s.append(np.std(factor_values))
    m = np.array(m)
    s = np.array(s)

    def generator():
        random_state = np.random.RandomState(random_seed)

        while True:
            # Note: remove batch dimension by returning x1[0], x2[0], y[0]
            factors = dset.sample_factors(
                1, random_state, **sample_factors_kwargs)
            x = dset.sample_observations_from_factors(factors, random_state)
            factors = (factors - m) / s  # normalize the factors
            y = factors * masks
            yield x[0], y[0]

    y_dim = masks.shape[-1]  # mask is 1-hot and equal in length to s_dim
    return tf.data.Dataset.from_generator(
        generator,
        (tf.float32, tf.float32),
        output_shapes=(dset.observation_shape, (y_dim,)))


def paired_randn(batch_size, z_dim, masks, mask_type="match"):
    if mask_type == "match":
        return match_randn(batch_size, z_dim, masks)
    elif mask_type == "rank":
        return rank_randn(batch_size, z_dim, masks)
    elif mask_type == "label":
        return label_randn(batch_size, z_dim, masks)


def match_randn(batch_size, z_dim, masks):
    # Note that masks.shape[-1] = s_dim and we assume s_dim <= z-dim
    n_dim = z_dim - masks.shape[-1]

    if n_dim == 0:
        z1 = tf.random_normal((batch_size, z_dim))
        z2 = tf.random_normal((batch_size, z_dim))
    else:
        # First sample the controllable latents
        z1 = tf.random_normal((batch_size, masks.shape[-1]))
        z2 = tf.random_normal((batch_size, masks.shape[-1]))

    # Do variable fixing here (controllable latents)
    mask_idx = tf.random_uniform(
        (batch_size,), maxval=len(masks), dtype=tf.int32)
    mask = tf.gather(masks, mask_idx)
    # mask = tf.expand_dims(tf.cast(mask, tf.float32), 1)
    z2 = z2 * mask + z1 * (1 - mask)

    # Add nuisance dims (uncontrollable latents)
    if n_dim > 0:
        z1_append = tf.random_normal((batch_size, n_dim))
        z2_append = tf.random_normal((batch_size, n_dim))
        z1 = tf.concat((z1, z1_append), axis=-1)
        z2 = tf.concat((z2, z2_append), axis=-1)

    return z1, z2, mask_idx


def rank_randn(batch_size, z_dim, masks):
    z1 = tf.random.normal((batch_size, z_dim))
    z2 = tf.random.normal((batch_size, z_dim))
    y = tf.gather(z1 > z2, masks, axis=-1)
    y = tf.cast(y, tf.float32)
    return z1, z2, y


def label_randn(batch_size, z_dim, masks):
    # Note that masks.shape[-1] = s_dim and we assume s_dim <= z-dim
    n_dim = z_dim - masks.shape[-1]

    if n_dim == 0:
        return tf.random.normal((batch_size, z_dim)) * (1 - masks)
    else:
        z = tf.random.normal((batch_size, masks.shape[-1])) * (1 - masks)
        n = tf.random.normal((batch_size, n_dim))
        z = tf.concat((z, n), axis=-1)
        return z
