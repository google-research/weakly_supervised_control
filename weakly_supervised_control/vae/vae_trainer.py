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
# from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
from torchvision.utils import save_image

from multiworld.core.image_env import normalize_image
from rlkit.core import logger
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

from weakly_supervised_control.vae.conv_vae import ConvVAE


class VAETrainer(ConvVAETrainer):
    def __init__(
            self,
            model: ConvVAE,
            train_dataset: np.ndarray,
            test_dataset: np.ndarray,
            train_factors: np.ndarray = None,
            test_factors: np.ndarray = None,
            pred_loss_coeff: float = 0.0,
            **kwargs,
    ):
        super().__init__(train_dataset=train_dataset,
                         test_dataset=test_dataset, model=model, **kwargs)
        self.train_factors = train_factors
        self.test_factors = test_factors
        self.pred_loss_coeff = pred_loss_coeff

    def get_batch(self, train=True, epoch=None, sample_factors: bool = False):
        dataset = self.train_dataset if train else self.test_dataset
        factors = self.train_factors if train else self.test_factors
        skew = False
        if epoch is not None:
            skew = (self.start_skew_epoch < epoch)
        if train and self.skew_dataset and skew:
            probs = self._train_weights / np.sum(self._train_weights)
            ind = np.random.choice(
                len(probs),
                self.batch_size,
                p=probs,
            )
        else:
            ind = np.random.randint(0, len(dataset), self.batch_size)

        sample_data = normalize_image(dataset[ind, :])

        if self.normalize:
            sample_data = ((sample_data - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            sample_data = sample_data - self.train_data_mean
        sample_data = ptu.from_numpy(sample_data)

        if sample_factors:
            return sample_data, ptu.from_numpy(factors[ind, :])
        else:
            return sample_data

    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        self.model.train()
        losses = []
        log_probs = []
        kles = []
        pred_losses = []
        zs = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(batches):
            self.optimizer.zero_grad()

            if self.pred_loss_coeff > 0:
                next_obs, factors = self.get_batch(
                    epoch=epoch, sample_factors=True)
                (reconstructions, obs_distribution_params,
                 latent_distribution_params, y_pred) = self.model(next_obs, predict_factors=True)
                pred_loss = self.model.prediction_loss(y_pred, factors)
                pred_losses.append(pred_loss.item())
            else:
                if sample_batch is not None:
                    data = sample_batch(self.batch_size, epoch)
                    next_obs = data['next_obs']
                else:
                    next_obs = self.get_batch(
                        epoch=epoch, sample_factors=False)
                (reconstructions, obs_distribution_params,
                 latent_distribution_params) = self.model(next_obs)
                pred_loss = 0

            log_prob = self.model.logprob(next_obs, obs_distribution_params)
            kle = self.model.kl_divergence(latent_distribution_params)
            encoder_mean = self.model.get_encoding_from_latent_distribution_params(
                latent_distribution_params)
            z_data = ptu.get_numpy(encoder_mean.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])

            loss = -1 * log_prob + beta * kle + self.pred_loss_coeff * pred_loss

            self.optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())

            self.optimizer.step()
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(next_obs)))
        if not from_rl:
            zs = np.array(zs)
            self.model.dist_mu = zs.mean(axis=0)
            self.model.dist_std = zs.std(axis=0)

        self.eval_statistics['train/log prob'] = np.mean(log_probs)
        self.eval_statistics['train/KL'] = np.mean(kles)
        self.eval_statistics['train/pred_loss'] = np.mean(pred_losses)
        self.eval_statistics['train/loss'] = np.mean(losses)

    # def get_diagnostics(self):
    #     return self.eval_statistics

    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_vae=True,
            from_rl=False,
            pred_loss_coeff: float = 0.0,
    ):
        self.model.eval()
        losses = []
        log_probs = []
        kles = []
        pred_losses = []
        zs = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(10):
            if pred_loss_coeff > 0:
                next_obs, factors = self.get_batch(
                    train=False, sample_factors=True)
                (reconstructions, obs_distribution_params,
                 latent_distribution_params, y_pred) = self.model(next_obs, predict_factors=True)
                pred_loss = self.model.prediction_loss(y_pred, factors)
                pred_losses.append(pred_loss.item())
            else:
                next_obs = self.get_batch(train=False, sample_factors=False)
                (reconstructions, obs_distribution_params,
                 latent_distribution_params) = self.model(next_obs)
                pred_loss = 0

            log_prob = self.model.logprob(next_obs, obs_distribution_params)
            kle = self.model.kl_divergence(latent_distribution_params)
            loss = -1 * log_prob + beta * kle + pred_loss_coeff * pred_loss

            encoder_mean = latent_distribution_params[0]
            z_data = ptu.get_numpy(encoder_mean.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())

            if batch_idx == 0 and save_reconstruction:
                n = min(next_obs.size(0), 8)
                comparison = torch.cat([
                    next_obs[:n].narrow(start=0, length=self.imlength, dim=1)
                    .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ).transpose(2, 3),
                    reconstructions.view(
                        self.batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize,
                    )[:n].transpose(2, 3)
                ])
                save_dir = osp.join(logger.get_snapshot_dir(),
                                    'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

        zs = np.array(zs)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['test/log prob'] = np.mean(log_probs)
        self.eval_statistics['test/KL'] = np.mean(kles)
        self.eval_statistics['test/pred_loss'] = np.mean(pred_losses)
        self.eval_statistics['test/loss'] = np.mean(losses)
        self.eval_statistics['beta'] = beta
        if not from_rl:
            for k, v in self.eval_statistics.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)
