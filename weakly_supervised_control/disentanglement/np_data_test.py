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
python -m weakly_supervised_control.disentanglement.np_dataset_test
"""
import time
import numpy as np

from weakly_supervised_control.disentanglement.np_data import NpGroundTruthData


def test_rank_pair():
    data = np.array([
        [1, 1, 1],
        [2, 2, 2],
    ])
    factors = np.array([
        [0, 3, 4],
        [1, 2, 5],
    ])
    num_factors = factors.shape[1]
    dset = NpGroundTruthData(data, factors)

    correct_rank_pairs = [
        ([1, 1, 1], [2, 2, 2], [1, 0, 1]),
        ([2, 2, 2], [1, 1, 1], [0, 1, 0]),
    ]
    for _ in range(10):
        stopwatch = time.time()
        masks = sorted(np.random.choice(num_factors, 2, replace=False))
        x1, x2, y = dset.sample_rank_pair(batch_size=1, masks=masks)
        print('Elapsed time for sample_rank_pair(): ', time.time() - stopwatch)

        is_correct = False
        for (correct_x1, correct_x2, rank) in correct_rank_pairs:
            correct_y = np.array(rank)[masks]
            if np.allclose(x1, correct_x1) and np.allclose(x2, correct_x2) and np.allclose(y, correct_y):
                is_correct = True
        assert is_correct, "Incorrect rank pair: x1 = {}, x2 = {}, y = {}, masks = {}".format(
            x1, x2, y, masks)


if __name__ == '__main__':
    test_rank_pair()
