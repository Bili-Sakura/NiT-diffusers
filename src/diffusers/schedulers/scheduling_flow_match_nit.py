# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover - importable without an installed diffusers checkout.
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
        pass

    def register_to_config(init):
        return init


@dataclass
class NiTFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class NiTFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Flow-matching ODE/SDE scheduler used by Native-resolution Image Synthesis (NiT).

    The model predicts velocity with a linear path by default. Timesteps run from 1 to 0,
    matching the original sampler while exposing the standard Diffusers `set_timesteps`
    and `step` API.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        mode: str = "ode",
        path_type: str = "linear",
        num_train_timesteps: int = 1000,
    ):
        if mode not in {"ode", "sde"}:
            raise ValueError("mode must be either 'ode' or 'sde'.")
        if path_type not in {"linear", "cosine"}:
            raise ValueError("path_type must be either 'linear' or 'cosine'.")
        self.mode = mode
        self.path_type = path_type
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(np.linspace(1.0, 0.0, num_train_timesteps + 1)).float()

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
        mode: Optional[str] = None,
    ):
        mode = mode or self.mode
        if mode == "sde":
            timesteps = torch.linspace(1.0, 0.04, num_inference_steps, dtype=torch.float32)
            timesteps = torch.cat([timesteps, torch.zeros(1, dtype=torch.float32)])
        elif mode == "ode":
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
        else:
            raise ValueError("mode must be either 'ode' or 'sde'.")
        self.mode = mode
        self.timesteps = timesteps.to(device=device)
        return self.timesteps

    @staticmethod
    def _expand_t_like_sample(timestep: torch.Tensor, sample: torch.Tensor, image_sizes: torch.LongTensor):
        dims = [1] * (sample.ndim - 1)
        seqlens = image_sizes[:, 0] * image_sizes[:, 1]
        if timestep.numel() == 1:
            timestep = timestep.repeat(image_sizes.shape[0])
        return torch.cat(
            [timestep[i].reshape(1, *dims).repeat(int(seqlens[i]), *dims) for i in range(image_sizes.shape[0])]
        )

    def _get_score_from_velocity(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        image_sizes: torch.LongTensor,
    ):
        timestep = self._expand_t_like_sample(timestep, sample, image_sizes)
        if self.path_type == "linear":
            alpha_t, d_alpha_t = 1 - timestep, torch.ones_like(timestep) * -1
            sigma_t, d_sigma_t = timestep, torch.ones_like(timestep)
        elif self.path_type == "cosine":
            alpha_t = torch.cos(timestep * np.pi / 2)
            sigma_t = torch.sin(timestep * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(timestep * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(timestep * np.pi / 2)
        else:
            raise ValueError(f"Unsupported path_type: {self.path_type}")
        reverse_alpha_ratio = alpha_t / d_alpha_t
        variance = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * model_output - sample) / variance

    @staticmethod
    def _compute_diffusion(timestep: torch.Tensor):
        return 2 * timestep

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        image_sizes: Optional[torch.LongTensor] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> NiTFlowMatchSchedulerOutput:
        del generator
        sample_dtype = sample.dtype
        sample = sample.to(torch.float64)
        model_output = model_output.to(torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()
        next_timestep = next_timestep.to(device=sample.device, dtype=torch.float64).flatten()

        if self.mode == "ode":
            prev_sample = sample + (next_timestep[0] - timestep[0]) * model_output
        else:
            if image_sizes is None:
                raise ValueError("image_sizes are required for SDE sampling.")
            image_sizes = image_sizes.to(device=sample.device, dtype=torch.long)
            diffusion = self._compute_diffusion(timestep[0])
            score = self._get_score_from_velocity(model_output, sample, timestep, image_sizes)
            drift = model_output - 0.5 * diffusion * score
            dt = next_timestep[0] - timestep[0]
            noise = torch.randn_like(sample)
            prev_sample = sample + drift * dt + torch.sqrt(diffusion) * noise * torch.sqrt(torch.abs(dt))

        prev_sample = prev_sample.to(sample_dtype)
        if not return_dict:
            return (prev_sample,)
        return NiTFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def step_heun(
        self,
        model_output: torch.Tensor,
        next_model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        next_timestep: torch.Tensor,
        return_dict: bool = True,
    ) -> NiTFlowMatchSchedulerOutput:
        if self.mode != "ode":
            raise ValueError("Heun correction is only defined for ODE sampling.")
        sample_dtype = sample.dtype
        sample = sample.to(torch.float64)
        model_output = model_output.to(torch.float64)
        next_model_output = next_model_output.to(torch.float64)
        timestep = timestep.to(device=sample.device, dtype=torch.float64).flatten()
        next_timestep = next_timestep.to(device=sample.device, dtype=torch.float64).flatten()
        prev_sample = sample + (next_timestep[0] - timestep[0]) * (0.5 * model_output + 0.5 * next_model_output)
        prev_sample = prev_sample.to(sample_dtype)
        if not return_dict:
            return (prev_sample,)
        return NiTFlowMatchSchedulerOutput(prev_sample=prev_sample)
