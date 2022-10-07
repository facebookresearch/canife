#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from hydra.utils import instantiate

from canife import Canary
from canife.utils import (
    clip_grad,
    compute_batch_grad,
    compute_local_update,
    compute_sample_grads,
    count_params,
    display_gpu_mem,
)


class CanaryDesigner():
    def __init__(self, grad_sample_module, canary_class=None, canary_loss="loss1", canary_norm_loss="hinge_squared", canary_design_type="sample_grads", canary_epochs=1000, 
                     canary_init="random", canary_preprocess=None, canary_clip_const=1, local_batch_size=128, canary_insert_batch_index=0, canary_design_local_models=False, 
                     server_clip_const=1, client_lr=1, num_classes=10, logger=None, local_updates=1, local_epochs=1, optimizer_config=None, dp_level="sample_level", 
                     gpu_mem_minimiser=False, canary_norm_matching=False, canary_norm_constant=50, canary_normalize_optim_grad=True,
                     in_channels=3, image_size=32, benchmark_design=False, **kwargs) -> None:

        self.canary_init = canary_init
        self.canary_loss = canary_loss
        self.canary_norm_loss = canary_norm_loss
        self.canary_norm_matching = canary_norm_matching
        self.canary_norm_constant = canary_norm_constant
        self.canary_normalize_optim_grad = canary_normalize_optim_grad
        
        self.canary_design_type = canary_design_type
        self.canary_class = canary_class
        self.canary_epochs = canary_epochs
        self.canary_clip_const = canary_clip_const

        self.canary_preprocess = canary_preprocess
        self.local_batch_size = local_batch_size
        self.canary_insert_batch_index = canary_insert_batch_index
        self.canary_design_local_models = canary_design_local_models
        self.canary_design_bias = 0

        self.canary_losses = canary_loss
        self.canary_type = "image"
        
        self.local_updates = local_updates
        self.local_epochs = local_epochs

        self.server_clip_const = server_clip_const
        self.client_lr = client_lr

        self.dp_level = dp_level
        self.num_classes = num_classes
        self.gpu_mem_minimiser = gpu_mem_minimiser
        
        self.logger = logger
        self.grad_sample_module = grad_sample_module
        self.optimizer_config = optimizer_config
        
        self.in_channels = in_channels
        self.image_size = image_size
        
        self.benchmark_design = benchmark_design
        self.benchmark_times = []

        # If user-level, design canary on unclipped gradients
        if self.dp_level == "user_level":
            self.canary_clip_const = float('inf')
    
    def get_analyser_args(self):
        """Returns attributes of CanaryDesigner which can be used to populate args when creating a CanaryAnalyser

        Returns:
            dict: attributes of CanaryDesigner
        """
        return self.__dict__

    def set_grad_sample_module(self, grad_sample_module):
        """
        Args:
            grad_sample_module (GradSampleModule): GradSampleModule to be used to compute per-sample gradients when designing the canary
        """
        self.grad_sample_module = grad_sample_module

    def _compute_clipped_grad(self, model, criterion, batch, device="cpu"):
        """Computes the clipped gradients of a batch

        Args:
            model: nn.Module to compute clipped grad
            criterion: Loss function
            batch: Batch to compute gradients from
            device (optional): Torch device. Defaults to "cpu".

        Returns:
            Clipped gradient of batch 
        """
        grad = compute_batch_grad(model, criterion, batch, device=device)
        # clip canary grad
        return clip_grad(grad, self.canary_clip_const)

    def _init_canary(self, canary_design_loader):
        """Initialises canary

        Args:
            canary_design_loader: Canary design pool, required for image initialisation

        Returns:
            canary: Canary as a tensor
        """
        if self.canary_init == "random":
            canary = torch.rand(size=(1,self.in_channels,self.image_size,self.image_size))
            canary = canary if self.canary_preprocess is None else self.canary_preprocess(canary)
            self.canary_class = random.randint(0, self.num_classes-1)
        else:
            if self.canary_design_type == "sample_grads": # The specific shapes of design loaders depends on sample_grads vs model_updates
                canary = next(iter(canary_design_loader))[0][0].clone().view(1,self.in_channels,self.image_size,self.image_size)
                self.canary_class = next(iter(canary_design_loader))[1][0].clone().item()
            else:
                canary = next(iter(canary_design_loader))[0][0][0].clone().view(1,self.in_channels,self.image_size,self.image_size)
                self.canary_class = next(iter(canary_design_loader))[0][1][0].clone().item()

        return canary.clone()
    
    def _compute_local_update(self, model, criterion, local_batches, device):
        """Computes a model update from a mock client who has local_batches

        Args:
            model: nn.Module
            criterion: Loss function
            local_batches: Clients local batches
            device: torch device

        Returns:
            model_update: An unscaled model update (clipped and then scaled by 1/lr * expected batch size)
        """
        initial_model_state = copy.deepcopy(model.state_dict())
        model_optimizer = instantiate(self.optimizer_config, model=model)
        local_model_state, local_model_before_insert, _ = compute_local_update(model, criterion, model_optimizer, local_batches, expected_batch_size=self.local_batch_size, local_epochs=self.local_epochs, reverse_batch_scaling=False, device=device)

        # Difference original and local model
        local_update = torch.tensor([]).to(device)
        for name, param in model.named_parameters():
            if param.requires_grad:
                local_update = torch.cat([local_update, (initial_model_state[name].data-local_model_state[name].data).flatten().detach().clone()])

        model.load_state_dict(initial_model_state) # Revert changes made by multiple local updates

        self.logger.debug(f"Mock client local update {local_update}, server clip const {self.server_clip_const}")

        # (1/lr)*B*clip(local update)
        # return (1/self.client_lr)*self.local_batch_size*clip_grad(local_update.cpu(), self.server_clip_const), local_model_before_insert
        return clip_grad(local_update.cpu(), self.server_clip_const), local_model_before_insert
    
    def _compute_aggregated_design_vectors(self, model, grad_dim, canary_design_loader, criterion, device):
        """Computes aggregated design vectors to craft canary on

        Args:
            model: nn.Module
            grad_dim: Gradient dimension of model
            canary_design_loader: Design loader
            criterion: Loss function
            device: torch device

        Returns:
            aggregated_design_vec: Either the aggregated sum of per-sample-gradients (if canary_design_type == sample_grads) or aggregated model updates (if canary_design_type == model_updates)
            batch_design_vecs: Individual per-sample gradients or individual model updates from mock design clients
            local_model_states: The final states of local models if canary_design_type=="model_updates"
        """
        aggregated_design_vec = torch.zeros(size=(grad_dim,))      
        batch_design_vecs = torch.tensor([])  
        local_model_states = []

        if self.canary_design_type == "sample_grads":
            batch_design_vecs = torch.zeros((grad_dim, ))
        elif self.canary_design_type == "model_updates":
            batch_design_vecs = torch.zeros((len(canary_design_loader), grad_dim))

        self.logger.info(" Computing sample grads/model updates from canary design pool...")
        for i, design_batch in enumerate(canary_design_loader):
            if i % 10 == 0:
                self.logger.debug(f" Computing sample grads/model updates of canary design batch={i+1}")

            if self.canary_design_type == "model_updates": # Scaled and clipped model updates
                local_update, before_insert_model_state, = self._compute_local_update(model, criterion, design_batch, device) # The design batch is a mock client's local daata
                batch_design_vecs[i] = local_update
                aggregated_design_vec += local_update
                local_model_states.append(before_insert_model_state)
                self.logger.debug(f"Mock client {i} scaled local update {local_update}")
                if i == 0:
                    self.logger.info(f"Local design updates are scaled by B={self.local_batch_size}, lr={self.client_lr}, clip const={self.server_clip_const}")
            elif self.canary_design_type == "gradient_pool":
                global_state = copy.deepcopy(self.grad_sample_module.state_dict())
                model_optimizer = instantiate(self.optimizer_config, model=self.grad_sample_module)
                _, _, local_step_sample_grads = compute_local_update(self.grad_sample_module, criterion, model_optimizer, design_batch, device=device, compute_sample_grads=True)
                self.grad_sample_module.load_state_dict(global_state) # Revert changes made by multiple local updates
                batch_design_vecs = torch.cat([batch_design_vecs, local_step_sample_grads], dim=0)
                aggregated_design_vec += local_step_sample_grads.sum(axis=0)
            else: 
                batch_design_vecs, _ = compute_sample_grads(self.grad_sample_module, criterion, design_batch, device=device, clipping_const=self.canary_clip_const)
                aggregated_design_vec += batch_design_vecs.sum(axis=0)

        return aggregated_design_vec, batch_design_vecs, local_model_states

    # Will be overriden for NLP
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
        
        base_lr = 1
        canary_optimizer = optim.Adam([canary], lr=base_lr)

        return init_canary, canary, canary_class, canary_optimizer
    
    # Will be overriden for NLP
    def _forward_pass_canary(self, model, canary):
        """Runs a forward pass on a canary given a model

        Args:
            model: nn.Module
            canary: canary tensor

        Returns:
            output: Output of model(canary)
        """
        model.train()
        model.zero_grad()
        output = model(canary)
        return output

    # Will be overriden for NLP
    def _post_process_canary(self, model, criterion, canary, canary_class, device="cpu"):
        """Computes final gradient from the canary

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
        canary_grad = self._compute_clipped_grad(model, criterion, [canary, canary_class], device=device).detach().cpu()
        return canary, canary_grad

    def _optimise(self, model, criterion, canary_design_loader, device="cpu"):
        """ Optimise over model and design loader to craft a canary

        Args:
            model: nn.Module
            criterion: Loss function
            canary_design_loader: DataLoader or list of tensors that mimics the batch structure of a DataLoader
            device (str, optional): Torch device, defaults to "cpu".

        Returns:
            canary: Canary object
        """
        display_gpu_mem(prefix="Start of optim", device=device, logger=self.logger)
        init_canary, canary, canary_class, canary_optimizer = self._init_canary_optimisation(canary_design_loader, device)
        model = model.to(device)
        model.zero_grad()
        
        # Init optim
        grad_dim = count_params(model)
        self.logger.info(f" Grad Dim {grad_dim}")
        canary_loss = torch.tensor(float("inf"))
        initial_model_state = copy.deepcopy(model.state_dict())
        local_model_states = []
        t, initial_canary_loss = 0,0
        optim_stats = defaultdict(list)
        best_canary = [float("inf"), None, 0]
        optim_improving = True
        aggregated_design_vec = torch.tensor([])
        x_grad_norm = 0

        display_gpu_mem(prefix="After moving model", device=device, logger=self.logger)

        # Compute the aggregated (sum or mean) grads of the canary design set and batch sample grads (if it fits into memory)
        if self.canary_loss == "loss1" or self.canary_design_sample_size <= self.canary_design_minibatch_size or self.canary_design_type != "sample_grads":
            aggregated_design_vec, batch_design_vecs, local_model_states = self._compute_aggregated_design_vectors(model, grad_dim, canary_design_loader, criterion, device)

        display_gpu_mem(prefix="After grad sample comp", device=device, logger=self.logger)

        self.logger.info("\n ===== Beginning canary optimization... =====")
        self.logger.info(f"Canary optimizer {canary_optimizer}")

        if self.canary_loss != "loss1" and (self.canary_design_sample_size <= self.canary_design_minibatch_size or self.canary_design_type != "sample_grads"): # i.e no minibatches
            target = batch_design_vecs # loss2 when sample grads fit into memory or when designing against model updates
            gradient_norms = torch.norm(target, dim=1)
            self.logger.info(f"Design norms {gradient_norms}") # Model updates or sample gradients
            self.logger.info(f"Average design norm {torch.mean(gradient_norms)}")
        else:
            target = aggregated_design_vec # loss1, optimisation target is the aggregated gradients or model updates

        display_gpu_mem(prefix="After target comp", device=device, logger=self.logger)

        parameters = []
        for p in model.parameters():
            if p.requires_grad:
                parameters.append(p)

        loss1_target = target.to(device)    
        epoch_target = loss1_target

        self.logger.debug(f"Pre-optim model arch {model}, {sum([p.flatten().sum() for p in model.parameters()])}")
        display_gpu_mem(prefix="Target moved to gpu", device=device, logger=self.logger)
        # grad_dim = 1

        model.zero_grad()
        while (t<=self.canary_epochs) and optim_improving:
            t+= 1
            if (t+1) % 100 == 0:
                loss_mean = np.mean(optim_stats["canary_loss"][-100:])
                self.logger.info(f" Canary optimisation, epoch={t}, initial loss={initial_canary_loss.item()}, average loss (last 100 iters)={loss_mean}, last loss={canary_loss.item()}")
            self.logger.debug(f" Canary grad (w.r.t canary loss) t={t}, {x_grad_norm}")

            if self.benchmark_design:
                start = timer()
                
            # Calculate loss of canary
            canary_optimizer.zero_grad()

            if (t+1) % 100 == 0 or t==1:
                display_gpu_mem(prefix=f"Start of optim t={t}", device=device, logger=self.logger)

            if len(local_model_states) > 0 and self.canary_insert_batch_index == -1 and self.canary_design_local_models:
                model.load_state_dict(local_model_states[random.randint(0, len(local_model_states)-1)]) # Randomly sample a local model to compute canary grad from

            if self.canary_loss == "loss2" and self.canary_design_sample_size > self.canary_design_minibatch_size or self.canary_loss == "loss_both": # minibatching
                if self.canary_design_type == "sample_grads": # Minibatch sample grads 
                    design_batch = next(iter(canary_design_loader))
                    epoch_target, _ = compute_sample_grads(self.grad_sample_module, criterion, design_batch, device, move_grads_to_cpu=False, clipping_const=self.canary_clip_const)  
                else: # Minibatch model updates 
                    idx = torch.ones(target.shape[0]).multinomial(num_samples=self.canary_design_minibatch_size, replacement=False).to(device)
                    epoch_target = target[idx]

            if (t+1) % 100 == 0 or t==1:
                display_gpu_mem(prefix=f"Minibatch optim t={t}", device=device, logger=self.logger)

            output = self._forward_pass_canary(model, canary) 
            loss = criterion(output, canary_class)

            self.logger.debug(f"Model canary {canary}, norm={torch.norm(canary)}")
            self.logger.debug(f"Model output {output}")
            self.logger.debug(f" Model loss t={t}, {loss}")
            canary_loss = torch.zeros(1, requires_grad=True).to(device)

            # hvp
            grad_f = autograd.grad(loss, parameters, create_graph=True, retain_graph=True)
            grad_f = torch.cat([g.flatten() for g in grad_f])
            self.logger.debug(f" Autograd grad_f t={t}, {grad_f}\n")
            self.logger.debug(f" Sum grad_f t={t}, {torch.sum(grad_f)}\n")
            temp_grad = grad_f.clone().detach().cpu()
            
            # Norm loss
            if self.canary_norm_matching and self.canary_norm_constant-torch.norm(grad_f) > 0:
                if self.canary_norm_loss == "hinge_squared":
                    canary_loss = canary_loss + grad_dim*((self.canary_norm_constant-torch.norm(grad_f)))**2
                else:
                    canary_loss = canary_loss + grad_dim*((self.canary_norm_constant-torch.norm(grad_f)))
            
            # Normalise canary grad
            if self.canary_normalize_optim_grad:
                grad_f = torch.nn.functional.normalize(grad_f, dim=0)*self.server_clip_const
                
            canary_loss = canary_loss + (grad_dim*(torch.sum(grad_f.view(1,-1) * epoch_target, dim=(1))**2).sum()/epoch_target.shape[0]) # Loss 1/2 term
            if self.canary_loss == "loss_both":
                canary_loss += (grad_dim*(torch.sum(grad_f.view(1,-1) * loss1_target, dim=(1))**2).sum()/loss1_target.shape[0])
                
            self.logger.debug(f" Canary loss t={t}, {canary_loss}\n")

            canary_loss.backward()
            canary_loss = canary_loss.detach().cpu()
            initial_canary_loss = canary_loss if t==1 else initial_canary_loss
            optim_stats["canary_loss"].append(canary_loss.item())
            optim_stats["canary_norm"].append(torch.norm(temp_grad).norm().item())
            x_grad_norm = torch.norm(canary.grad.detach()).cpu()

            if (t+1) % 100 == 0 or t==1:
                display_gpu_mem(prefix=f"Pre-end of optim t={t}", device=device, logger=self.logger)

            if t < self.canary_epochs:
                canary_optimizer.step()
            model.zero_grad()

            # if canary_loss < best_canary[0]:
            if True and t == self.canary_epochs:
                best_canary = [canary_loss.detach().cpu(), canary.detach().clone().cpu(), t]
            
            if (t+1) % 100 == 0 or t==1:
                display_gpu_mem(prefix=f"End of optim t={t}", device=device, logger=self.logger)

            if self.benchmark_design:
                end = timer()
                self.benchmark_times.append(end-start)
                
        best_canary_loss, canary, best_t = best_canary

        # Computes grad of canary from the model 
        # For NLP this will sample the canary and compute the exact gradient 
        canary, canary_grad = self._post_process_canary(model, criterion, canary, canary_class, device=device)
        init_canary, init_canary_grad = self._post_process_canary(model, criterion, init_canary, canary_class, device=device)

        self.logger.debug(f"Clipped gradient computed {torch.sum(canary_grad)}, {canary_grad}")

        self.logger.info(f" Grad Descent for canary...t={t}")
        self.logger.info(f" Best canary at t={best_t}, {best_canary_loss}")
        canary_health = ((initial_canary_loss-best_canary_loss) / initial_canary_loss).item()
        self.logger.info(f" Canary Norm {torch.norm(canary_grad).item()}")
        self.logger.info(f" Canary Health {canary_health}")

        if self.canary_loss == "loss1" or self.canary_design_sample_size <= self.canary_design_minibatch_size or self.canary_design_type == "model_updates":
            aggregated_design_vec = aggregated_design_vec/self.canary_design_pool_size
            self.canary_design_bias = -torch.dot(canary_grad/torch.norm(canary_grad), aggregated_design_vec).cpu().detach().item()
            self.logger.info(f"Canary grad {canary_grad}")
            self.logger.info(f"Canary grad normalised {canary_grad/torch.norm(canary_grad)}")
            self.logger.info(f"Dot Product <Canary/||grad(can)||, S> {-self.canary_design_bias}")
            self.logger.info(f"Dot Product <Canary/||grad(can)||, S+canary> {torch.dot(canary_grad/torch.norm(canary_grad), aggregated_design_vec + (canary_grad/torch.norm(canary_grad))).cpu().detach().item()}")
            self.logger.info(f"Canary batch gradients {aggregated_design_vec + canary_grad/torch.norm(canary_grad)}")

        self.logger.info(f" x.grad Norm {x_grad_norm}\n\n")
        
        self.canary_losses = optim_stats["canary_loss"]
        self.canary_norms = optim_stats["canary_norm"]

        model.load_state_dict(initial_model_state) 

        return Canary(canary, init_canary, canary_class.item(), init_loss=initial_canary_loss.item(), init_grad=init_canary_grad, 
                        final_loss=best_canary_loss.item(), canary_grad=canary_grad, health=canary_health)

    def _update_design_params(self, canary_design_loader, clients_per_round, design_minibatch_size=None, varying_local_batches=False):
        """Updates relevant design params (canary_design_sample_size, canary_design_pool_size, canary_design_minibatch_size) 
            will infer this from the canary_design_loader and other provided args

        Args:
            canary_design_loader: Design loader
            design_minibatch_size (optional): To override inferred minibatch size. Defaults to None which sets design_minibatch_size to num_local_updates
            varying_local_batches (bool): If True then clients have varying batch sizes. Defaults to False.
        """
        example_design_batch = next(iter(canary_design_loader))[0] if self.canary_design_type == "sample_grads" else canary_design_loader[0] # Either a batch of sample gradients or a mock client
        num_local_updates = -1
        
        if self.canary_design_type == "sample_grads":
            self.canary_design_minibatch_size = example_design_batch.shape[0]
            self.local_batch_size = self.canary_design_minibatch_size
            self.canary_design_sample_size = len(canary_design_loader) * self.canary_design_minibatch_size
            self.canary_design_pool_size = self.canary_design_sample_size
        else:
            if not varying_local_batches:
                self.local_batch_size = example_design_batch[0][0].shape[0]
                num_local_updates = len(example_design_batch)

            self.canary_design_minibatch_size = design_minibatch_size if design_minibatch_size else clients_per_round
            self.canary_design_sample_size = sum([sum([batch[0].shape[0] for batch in mock_client]) for mock_client in canary_design_loader])
            self.canary_design_pool_size = len(canary_design_loader)

            if self.canary_design_type == "gradient_pool":
                self.canary_design_pool_size = self.canary_design_sample_size

            if self.canary_design_type == "model_updates" and self.canary_design_minibatch_size > self.canary_design_pool_size:
                self.canary_design_minibatch_size = self.canary_design_pool_size
        self.logger.info(f"Designer inferred design sample size={self.canary_design_sample_size}, design pool={self.canary_design_pool_size}, minibatch size={self.canary_design_minibatch_size}, local updates={num_local_updates}, local client batch size={self.local_batch_size}")

    def design(self, model, criterion, canary_design_loader, clients_per_round=100, varying_local_batches=False, canary_design_minibatch_size=None, device="cpu"):
        """Designs a canary from a given model and design pool (canary_design_loader)

        Args:
            model: nn.Module
            criterion: Loss function
            canary_design_loader: Design loader
            varying_local_batches (bool, optional): If True, design clients contain varying batch sizes. Defaults to False.
            canary_design_minibatch_size (optional): Minibatch size for designing. Defaults to None.
            device (optional): Torch device to design on, defaults to "cpu".

        Returns:
            canary: Canary object
        """
        assert self.grad_sample_module is not None, "Must set_grad_sample_module before designing a canary"
                
        display_gpu_mem(prefix="Start of design", device=device, logger=self.logger) # For debugging
        self.grad_sample_module.to(device)
        display_gpu_mem(prefix="Grad sample module moved", device=device, logger=self.logger) # For debugging

        self.logger.debug(f"Design model arch {model}, {sum([p.flatten().sum() for p in model.parameters()])}")

        # Infer design parameters such as the design pool + sample size from the canary_design_loader
        self._update_design_params(canary_design_loader, clients_per_round, design_minibatch_size=canary_design_minibatch_size, varying_local_batches=varying_local_batches)

        # Optimise and find canary
        canary = self._optimise(model, criterion, canary_design_loader, device)

        # To avoid GPU mem issues with FLSim if using GPUMemoryMinimiser
        if self.gpu_mem_minimiser:
            self.grad_sample_module.to("cpu") 
            model.to("cpu")

        return canary