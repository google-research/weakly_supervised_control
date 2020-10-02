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
"""Tensorsketch API.
"""


from weakly_supervised_control.disentanglement.tensorsketch.modules import Module
from weakly_supervised_control.disentanglement.tensorsketch.modules import ModuleList
from weakly_supervised_control.disentanglement.tensorsketch.modules import Sequential

from weakly_supervised_control.disentanglement.tensorsketch.modules import Affine
from weakly_supervised_control.disentanglement.tensorsketch.modules import Dense
from weakly_supervised_control.disentanglement.tensorsketch.modules import Conv2d
from weakly_supervised_control.disentanglement.tensorsketch.modules import ConvTranspose2d

from weakly_supervised_control.disentanglement.tensorsketch.modules import Flatten
from weakly_supervised_control.disentanglement.tensorsketch.modules import Reshape

from weakly_supervised_control.disentanglement.tensorsketch.modules import ReLU
from weakly_supervised_control.disentanglement.tensorsketch.modules import LeakyReLU
from weakly_supervised_control.disentanglement.tensorsketch.modules import Sigmoid

from weakly_supervised_control.disentanglement.tensorsketch.normalization import BatchNorm
from weakly_supervised_control.disentanglement.tensorsketch.normalization import SpectralNorm
from weakly_supervised_control.disentanglement.tensorsketch.normalization import WeightNorm

from weakly_supervised_control.disentanglement.tensorsketch.utils import advanced_function
from weakly_supervised_control.disentanglement.tensorsketch.utils import reset_tf_function
