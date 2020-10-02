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
"""Modules API.
"""


from weakly_supervised_control.disentanglement.tensorsketch.modules.base import Module
from weakly_supervised_control.disentanglement.tensorsketch.modules.base import ModuleList
from weakly_supervised_control.disentanglement.tensorsketch.modules.base import Sequential

from weakly_supervised_control.disentanglement.tensorsketch.modules.shape import Flatten
from weakly_supervised_control.disentanglement.tensorsketch.modules.shape import Reshape

from weakly_supervised_control.disentanglement.tensorsketch.modules.affine import Affine
from weakly_supervised_control.disentanglement.tensorsketch.modules.affine import Dense
from weakly_supervised_control.disentanglement.tensorsketch.modules.affine import Conv2d
from weakly_supervised_control.disentanglement.tensorsketch.modules.affine import ConvTranspose2d

from weakly_supervised_control.disentanglement.tensorsketch.modules.activation import ReLU
from weakly_supervised_control.disentanglement.tensorsketch.modules.activation import LeakyReLU
from weakly_supervised_control.disentanglement.tensorsketch.modules.activation import Sigmoid
