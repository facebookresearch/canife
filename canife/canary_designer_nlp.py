#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from canife import CanaryDesigner
from canife.utils import TextProcessorSent140, TextProcessorShakes


class CanaryDesignerNLP(CanaryDesigner):
    def __init__(self, grad_sample_module, canary_class=None, canary_loss="loss1", canary_norm_loss="hinge_squared", canary_design_type="sample_grads", canary_epochs=1000, canary_init="random", canary_preprocess=None, canary_clip_const=1, 
                     local_batch_size=128, canary_insert_batch_index=0, canary_design_local_models=False, server_clip_const=1, client_lr=1, 
                     num_classes=10, logger=None, local_updates=1, local_epochs=1, optimizer_config=None, dp_level="sample_level", gpu_mem_minimiser=False,
                     canary_norm_matching=False, canary_norm_constant=50, canary_normalize_optim_grad=True,
                     benchmark_design=False, **kwargs) -> None:

        super().__init__(grad_sample_module=grad_sample_module, canary_class=canary_class, canary_loss=canary_loss, canary_norm_loss=canary_norm_loss, canary_design_type=canary_design_type, canary_epochs=canary_epochs, 
                        canary_init=canary_init, canary_preprocess=canary_preprocess, canary_clip_const=canary_clip_const, local_batch_size=local_batch_size, canary_insert_batch_index=canary_insert_batch_index, 
                        canary_design_local_models=canary_design_local_models, server_clip_const=server_clip_const, client_lr=client_lr, 
                        num_classes = num_classes, logger=logger, local_updates=local_updates, local_epochs=local_epochs, optimizer_config=optimizer_config, dp_level=dp_level, gpu_mem_minimiser=gpu_mem_minimiser, canary_norm_matching=canary_norm_matching, 
                        canary_norm_constant=canary_norm_constant, canary_normalize_optim_grad=canary_normalize_optim_grad, benchmark_design=benchmark_design, **kwargs)

        self.text_processor = TextProcessorShakes() if kwargs["dataset"] == "shakespeare" else TextProcessorSent140()
        self.canary_type = "nlp"

    def _init_canary_optimisation(self, canary_design_loader, device):
        """Initialises canaries for optimisation

        Args:
            canary_design_loader: Design pool
            device: Torch device

        Returns:
            init_canary: Initial Canary for metrics
            canary: Tensor canary to optimise
            canary_class: Tensor class of canary 
            canary_optimizer: Optimizer over canary
        """
        init_canary = self._init_canary(canary_design_loader)
        canary = init_canary.clone().to(device) # Clone because we keep the initial canary for statistics
        canary.requires_grad = True
        canary_class = torch.tensor([self.canary_class]).to(device)
        canary_optimizer = torch.optim.Adam([canary], lr=0.1)

        return init_canary, canary, canary_class, canary_optimizer
        
    def _init_canary(self, canary_design_loader):
        """Initialises canary

        Args:
            canary_design_loader: Canary design pool, required to infer sequence length for text initialisation

        Returns:
            canary: Canary as a tensor
        """
        # Initialise log coeffs
        if self.canary_design_type == "sample_grads":
            example_seq = next(iter(canary_design_loader))[0][0].clone()
            self.canary_class = next(iter(canary_design_loader))[1][0].clone().item()
        else:
            example_seq = next(iter(canary_design_loader))[0][0][0].clone()
            self.canary_class = next(iter(canary_design_loader))[0][1][0].clone().item()

        if self.canary_init == "random":
            log_coeffs = torch.rand(len(example_seq), self.text_processor.vocab_size)
            self.canary_class = random.randint(0, self.num_classes-1)
        else:
            log_coeffs = torch.zeros(len(example_seq), self.text_processor.vocab_size)
            indices = torch.arange(log_coeffs.size(0)).long()
            log_coeffs[indices, example_seq] = 12

        self.logger.info(f"Log coeffs initialised shape={log_coeffs.shape}")
        return log_coeffs
        
    def _forward_pass_canary(self, model, canary):
        """Runs a forward pass on a canary given a model
        
        Uses the Gumbel softmax method of Guo et al. (2021) (https://arxiv.org/abs/2104.13733)

        Args:
            model: nn.Module
            canary: canary tensor

        Returns:
            output: Output of model(canary)
        """
        model.train()
        model.zero_grad()

        # Gumbel softmax the log coeffs
        coeffs = F.gumbel_softmax(canary, hard=False) # T x V

        # Form soft embeddings
        embedding_weights = model.__dict__["_modules"]["embedding"].weight

        inputs_embeds = (coeffs @ embedding_weights) # T x D

        # Forward pass through model (using soft embeddings as input)
        pred = model(None, input_embeds=inputs_embeds.unsqueeze(0))

        return pred

    def _post_process_canary(self, model, criterion, canary, canary_class, device="cpu"):
        """Computes final gradient from the canary. Converts token distribution to text sample

        Args:
            model: nn.Module
            criterion: Loss function
            canary: tensor
            canary_class: tensor
            device (optional): torch device, defaults to "cpu".

        Returns:
            canary: Final canary after post-processsing
            canary_grad: Final canary gradient
        """
        # self._plot_canary_dist(canary)
        canary = F.gumbel_softmax(canary, hard=True).argmax(1).unsqueeze(0).long()
        canary_grad = self._compute_clipped_grad(model, criterion, [canary, canary_class], device=device).detach().cpu()
        return canary, canary_grad

    def _plot_canary_dist(self, canary):
        """
        For debugging. Plots the token distribution of the canary.
        
        Args:
            canary: canary token distribution to plot
        """
        coeffs = F.gumbel_softmax(canary, hard=False)

        for row in coeffs:
            row = np.array(row)
            sns.barplot(x=list(range(0, len(row))), y=row)
            plt.plot()
            plt.pause(1)
            plt.clf()
    