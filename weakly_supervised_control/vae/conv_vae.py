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
from typing import List
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
#from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.torch.vae.conv_vae import ConvVAE

imsize48_default_architecture = dict(
    conv_args=dict(
        kernel_sizes=[5, 3, 3],
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
    ),
    conv_kwargs=dict(
        hidden_sizes=[],
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,
    )
)


class VAE(ConvVAE):
    def __init__(
            self,
            x_shape: List[int],
            representation_size: int,
            init_w: float = 1e-3,
            num_factors: int = None,
            **kwargs
    ):
        """
        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        architecture = imsize48_default_architecture
        if len(x_shape) == 3:
            architecture['deconv_output_channels'] = x_shape[-1]
        else:
            assert len(x_shape) == 4
            # TODO: Implement siamese network.
            raise NotImplementedError()

        super().__init__(representation_size=representation_size,
                         architecture=architecture, init_w=init_w, **kwargs)

        if num_factors is not None:
            self.fc3 = nn.Linear(representation_size,
                                 num_factors)  # factor prediction
            self.fc3.weight.data.uniform_(-init_w, init_w)
            self.fc3.bias.data.uniform_(-init_w, init_w)
        else:
            self.fc3 = None

    def predict_factors(self, input):
        mu, _ = self.encode(input)
        return F.sigmoid(self.fc3(mu))

    def prediction_loss(self, pred, labels):
        return F.mse_loss(pred, labels, reduction='elementwise_mean')

    def forward(self, input, predict_factors=False):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        reconstructions, obs_distribution_params, latent_distribution_params = super().forward(input)

        if predict_factors:
            y_pred = self.predict_factors(input)
            return reconstructions, obs_distribution_params, latent_distribution_params, y_pred
        else:
            return reconstructions, obs_distribution_params, latent_distribution_params
