# Adapted from MONAI generative models:
#
# Copyright (c) MONAI Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

import numpy as np
import torch

from net_utils.scheduler import Scheduler
from net_utils.simplex_noise import generate_noise


class DDPMScheduler(Scheduler):
    """
    Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling. Based on: Ho et al., "Denoising Diffusion Probabilistic Models"
    https://arxiv.org/abs/2006.11239

    Args:
        num_train_timesteps: number of diffusion steps used to train the model.
        beta_start: the starting `beta` value of inference.
        beta_end: the final `beta` value.
        beta_schedule: {``"linear"``, ``"scaled_linear"``}
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model.
        variance_type: {``"fixed_small"``, ``"fixed_large"``, ``"learned"``, ``"learned_range"``}
            options to clip the variance used when adding noise to the denoised sample.
        clip_sample: option to clip predicted sample between -1 and 1 for numerical stability.
        prediction_type: {``"epsilon"``, ``"sample"``, ``"v_prediction"``}
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
    """

    def __init__(self, variance_type: str = "fixed_small", **kwargs) -> None:
        super().__init__(**kwargs)
        if self.prediction_type.lower() not in ["epsilon", "sample", "v_prediction"]:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or" " `v_prediction`"
            )
        self.variance_type = variance_type

    def get_timesteps(self, noise_level: int):
        """
        Returns the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            noise_level: number of diffusion steps used when generating samples with a pre-trained model.

        Returns:
            Returns the timesteps
        """
        if noise_level > self.num_train_timesteps:
            raise ValueError(
                f"`noise_level`: {noise_level} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        timesteps = torch.from_numpy(np.arange(0, noise_level + 1)[::-1].copy())
        return timesteps

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain. Initialises the timesteps attribute.
        
        Args:
            num_inference_steps: number of diffusion steps used when generating samples with a pre-trained model.
            device: target device to put the data.
        """
        assert num_inference_steps == self.num_train_timesteps, "{self.__class__} can only do one step at a time like in the forward process."

        self.num_inference_steps = self.num_train_timesteps
        self.timesteps = self.get_timesteps(1000).to(device)

    def _get_mean(self, timestep: int, x_0: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of the posterior at timestep t.

        Args:
            timestep: current timestep.
            x0: the noise-free input.
            x_t: the input noised to timestep t.

        Returns:
            Returns the mean
        """
        # these attributes are used for calculating the posterior, q(x_{t-1}|x_t,x_0),
        # (see formula (5-7) from https://arxiv.org/pdf/2006.11239.pdf)
        alpha_t = self.alphas[timestep]
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)

        x_0_coefficient = alpha_prod_t_prev.sqrt() * self.betas[timestep] / (1 - alpha_prod_t)
        x_t_coefficient = alpha_t.sqrt() * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)

        mean = x_0_coefficient * x_0 + x_t_coefficient * x_t

        return mean

    def _get_variance(self, timestep: int, predicted_variance: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the variance of the posterior at timestep t.

        Args:
            timestep: current timestep.
            predicted_variance: variance predicted by the model.

        Returns:
            Returns the variance
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[timestep]
        # hacks - were probably added for training stability
        if self.variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        elif self.variance_type == "fixed_large":
            variance = self.betas[timestep]
        elif self.variance_type == "learned":
            return predicted_variance
        elif self.variance_type == "learned_range":
            min_log = variance
            max_log = self.betas[timestep]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.
            generator: random number generator.

        Returns:
            pred_prev_sample: Predicted previous sample
        """
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if timestep > 0:
            noise = generate_noise(self.noise_type, model_output, timestep)
            variance = (self._get_variance(timestep, predicted_variance=predicted_variance) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample
