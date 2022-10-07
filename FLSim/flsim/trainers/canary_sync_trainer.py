#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import math
import random
import sys
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flsim.channels.message import Message
from flsim.clients.base_client import Client
from flsim.clients.dp_client import DPClient
from flsim.common.timeline import Timeline
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, Metric, TrainingStage
from flsim.interfaces.model import IFLModel
from flsim.servers.sync_servers import ISyncServer, SyncServerConfig
from flsim.trainers.sync_trainer import SyncTrainer
from flsim.trainers.trainer_base import FLTrainerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.cuda import GPUMemoryMinimizer
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from flsim.utils.fl.stats import RandomVariableStatsTracker
from hydra.utils import instantiate
from omegaconf import OmegaConf
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import normalize
from tqdm import tqdm


sys.path.append("../../mad_canaries")
from canife import CanaryAnalyser, CanaryDesigner, CanaryDesignerNLP
from canife.utils import display_gpu_mem


class CanarySyncTrainer(SyncTrainer):
    """Implements synchronous Federated Learning Training.

    Defaults to Federated Averaging (FedAvg): https://arxiv.org/abs/1602.05629
    """

    def __init__(
        self,
        *,
        model: IFLModel,
        cuda_enabled: bool = False,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=CanarySyncTrainerConfig,
            **kwargs,
        )

        super().__init__(model=model, cuda_enabled=cuda_enabled, **kwargs)

        if self.cfg.args.gpu_mem_minimiser:
            self._cuda_state_manager = GPUMemoryMinimizer(cuda_enabled)
            self._cuda_state_manager.on_trainer_init(model)

        self.server: ISyncServer = instantiate(
            # pyre-ignore[16]
            self.cfg.server,
            global_model=model,
            channel=self.channel,
        )
        self.clients = {}
        self.mock_clients = {} # For canary design
        self.accuracy_metrics = {"train": [], "agg": [], "eval": [], "test": []}
        self.insert_acc_achieved = False
        self.initial_privacy_budget = None # For checkpointing priv budget
        self.one_step_budget = None # For comparing empirical epsilon with single step epsilon

    def global_model(self) -> IFLModel:
        """self.global_model() is owned by the server, not by SyncTrainer"""
        return self.server.global_model

    def create_or_get_client_for_data(self, dataset_id: int, datasets: Any):
        """Creates clients for a round. This function is called UPR * num_rounds
        times per training run. Here, we use <code>OmegaConf.structured</code>
        instead of <code>hydra.instantiate</code> to minimize the overhead of
        hydra object creation.
        """
        if self.is_sample_level_dp:
            client = DPClient(
                # pyre-ignore[16]
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_train_user(dataset_id),
                name=f"client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        else:
            client = Client(
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_train_user(dataset_id),
                name=f"client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        # self.clients[dataset_id] = client  
        # return self.clients[dataset_id]
        return self.clients.get(dataset_id, client)

    def _create_mock_client(self, dataset_id: int, datasets: Any):
        """
        """
        if self.is_sample_level_dp:
            client = DPClient(
                # pyre-ignore[16]
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_test_user(dataset_id),
                name=f"mock_client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        else:
            client = Client(
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_test_user(dataset_id),
                name=f"mock_client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        return self.mock_clients.get(dataset_id, client)

    def _get_checkpoint_path(self, epoch, round, final=False):
        canary_type = self.cfg.args.canary_design_type
        filename =  self.cfg.args.checkpoint_path

        if final:
            filename += f"/FLSim_dp={self.cfg.args.dp_level}_model={self.cfg.args.model_arch}_dataset={self.cfg.args.dataset}_num_clients={self.cfg.args.users_per_round}_test_size={self.cfg.args.local_batch_size}"
            filename += f"_insert_test_acc={self.cfg.args.canary_insert_test_acc}_insert_train_acc={self.cfg.args.canary_insert_train_acc}_client_epochs={self.cfg.args.client_epochs}"
        else:
            filename += "/" + "FLSim_" + self.cfg.args.dp_level + "_epoch=" +str(epoch) + "_round=" + str(round) + "_" + self.cfg.args.model_arch + "_" + canary_type

        if self.cfg.args.epsilon != -1 or self.cfg.args.sigma != 0:
            if self.cfg.args.epsilon != -1:
                filename += f"_private_eps={self.cfg.args.epsilon}_delta={self.cfg.args.delta}"
            else:
                filename += f"_private_sigma={self.cfg.args.sigma}_delta={self.cfg.args.delta}"
                
        filename += ".tar"
        return filename

    def _checkpoint_model(self, epoch, round, final=False, mock_client_indices=None):
        privacy_budget = self.server._privacy_engine.get_privacy_spent()
        train_acc = self.accuracy_metrics["train"][-1] if len(self.accuracy_metrics["train"]) > 0 else 0
        test_acc = self.accuracy_metrics["test"][-1] if len(self.accuracy_metrics["test"]) > 0 else 0

        checkpoint = {
                "epoch": epoch,
                "round": round,
                "noise_multiplier": self.server._privacy_engine.noise_multiplier,
                "epsilon": privacy_budget.epsilon,
                "delta": privacy_budget.delta,
                "steps": self.server._privacy_engine.steps,
                "state_dict": self.global_model().fl_get_module().state_dict(),
                "mock_client_indices": mock_client_indices,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }

        checkpoint_path = self._get_checkpoint_path(epoch, round, final=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Client selection
    def _client_selection(
        self,
        num_users: int,
        users_per_round: int,
        data_provider: IFLDataProvider,
        global_model: IFLModel,
        epoch: int,
        clients_to_exclude: list,
    ) -> List[Client]:
        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        num_users_overselected = math.ceil(users_per_round / self.cfg.dropout_rate)
        user_indices_overselected = []

        user_indices_overselected = self.server.select_clients_for_training(
            num_total_users=num_users-len(clients_to_exclude),
            users_per_round=num_users_overselected,
            data_provider=data_provider,
            epoch=epoch,
        )
        
        for i, selected_client in enumerate(user_indices_overselected):
            for j, excluded_client in enumerate(clients_to_exclude):
                if selected_client == excluded_client:
                    user_indices_overselected[i] = num_users-len(clients_to_exclude) + j

        clients_triggered = [
            self.create_or_get_client_for_data(i, self.data_provider)
            for i in user_indices_overselected
        ]

        clients_to_train = self._drop_overselected_users(
            clients_triggered, users_per_round
        )

        return clients_to_train

    def _select_mock_client_indices(self, num_users, users_per_round):
        num_users = self.data_provider.num_test_users()
        batches_per_client, samples_per_batch = -1,-1
        
        if self.cfg.args.canary_design_type == "model_updates" and self.cfg.args.canary_design_pool_size > 0:
            num_mock_clients = self.cfg.args.canary_design_pool_size
            mock_client_indices = np.random.choice(range(0, num_users), num_mock_clients, replace=False)
        else:
            shuffled_clients = np.random.permutation(range(0, num_users))
            total_samples = 0

            for i, client_id in enumerate(shuffled_clients):
                client = self._create_mock_client(client_id, self.data_provider)
                total_samples += client.dataset.num_eval_examples()
                if total_samples >= self.cfg.args.canary_design_sample_size:
                    break

            mock_client_indices = shuffled_clients[:i+1]

        self.logger.info(f"Design Pool Size: {self.cfg.args.canary_design_pool_size}, Number of mock clients needed: {len(mock_client_indices)}")
        self.logger.info(f"Local updates per client {batches_per_client}, Samples per update {samples_per_batch}")

        return mock_client_indices, batches_per_client, samples_per_batch

    def _select_mock_clients(self, client_indices):
        return [self._create_mock_client(client_idx, self.data_provider) for client_idx in client_indices]
    
    # Designing and Pushing the canary
    def _create_canary_design_loader(self, mock_clients, exact_testing=False):
        # If the design requires model updates, batch per design client
        if self.cfg.args.canary_design_type != "sample_grads":
            design_loader = []
            for i, mock_client in enumerate(mock_clients):
                if exact_testing:
                    local_batch = [(batch["features"], batch["labels"].long()) for batch in mock_client.dataset._user_batches] # Note: If exact, mock clients are real clients from train set
                else:
                    local_batch = [(batch["features"], batch["labels"].long()) for batch in mock_client.dataset._eval_batches] # Note: Mock clients are from the test set so only contain _eval_batches (not _user_batches)
                design_loader.append(local_batch)
            self.logger.info(f"Design Loader {len(design_loader)}")
        else:
            if not exact_testing:
                design_x = torch.vstack([torch.vstack([mock_client.dataset._eval_batches[i]["features"] for i in range(len(mock_client.dataset._eval_batches))]) for mock_client in mock_clients])
                design_y = torch.hstack([torch.hstack([mock_client.dataset._eval_batches[i]["labels"] for i in range(len(mock_client.dataset._eval_batches))]) for mock_client in mock_clients]).long()
                design_x = design_x[:self.cfg.args.canary_design_sample_size]
                design_y = design_y[:self.cfg.args.canary_design_sample_size]
                batch_size = self.cfg.args.canary_design_minibatch_size
            else:
                design_x = torch.vstack([torch.vstack([mock_client.dataset._user_batches[i]["features"] for i in range(len(mock_client.dataset._user_batches))]) for mock_client in mock_clients])
                design_y = torch.hstack([torch.hstack([mock_client.dataset._user_batches[i]["labels"] for i in range(len(mock_client.dataset._user_batches))]) for mock_client in mock_clients]).long()
                batch_size = design_x.shape[0]

            design_loader = DataLoader(TensorDataset(design_x, design_y), batch_size=batch_size , shuffle=True) 
            self.logger.info(f"Canary Design Pool x {design_x.shape}, Canary Design Pool y  {design_y.shape}")

        return design_loader

    def _push_canary_to_client(self, clients, canary, target_batch_idx=-1, insert_as_batch=False, type="canary"):

        canary_data = canary.data if type=="canary" else canary.init_data
        canary_class = canary.class_label

        # select target client at random
        target_client = random.choice(clients)
        if self.cfg.args.canary_setup == "exact":
            target_client = clients[0] # For debugging

        self.logger.info(f"Inserting canary to client {target_client.name}")
        
        if insert_as_batch: # replace the batch with a single canary sample
            old_data = copy.deepcopy(target_client.dataset._user_batches)
            target_client.dataset._user_batches= [{"features": canary_data, "labels": torch.tensor(canary_class).unsqueeze(0)}]
        else: # insert canary as the first sample in the target_batch_idx batch
            # test canary shape 
            batch_sizes =  target_client.dataset._user_batches[target_batch_idx]["features"].shape 
            assert canary_data.shape[1:] == batch_sizes[1:], "Wrong canary size"

            # test target batch idx 
            num_batches = len(target_client.dataset._user_batches)
            assert target_batch_idx < num_batches, "Wrong target_batch_idx"

            old_x, old_y = target_client.dataset._user_batches[target_batch_idx]["features"][0].clone(), target_client.dataset._user_batches[target_batch_idx]["labels"][0].clone()
            old_data = (old_x, old_y)
            target_client.dataset._user_batches[target_batch_idx]["features"][0] = canary_data
            target_client.dataset._user_batches[target_batch_idx]["labels"][0] = canary_class

        target_client.flag_as_canary_inserted() # Forces canary client to take a single local epoch
        return target_client, old_data

    def _design_canary_from_mock_clients(self, mock_clients, canary_designer, model=None, exact_testing=False):
        """
        Generates a canary from a clients local dataset using a CanaryDesigner

        Args:
            - mock_clients: List of clients to design the canary from (will be excluded from training)
            - canary_designer: CanaryDesigner
        """
        # Create design batches from chosen mock client(s)
        self.logger.info(f"Designing canary from mock clients {[client.name for client in mock_clients]}")
        canary_design_loader = self._create_canary_design_loader(mock_clients, exact_testing=exact_testing)        

        # Craft canary
        model = self.global_model().fl_get_module() if model is None else model
        canary = canary_designer.design(model, F.cross_entropy, canary_design_loader, clients_per_round=self.cfg.users_per_round, canary_design_minibatch_size=self.cfg.args.canary_design_minibatch_size, varying_local_batches=(self.cfg.args.num_local_updates==-1), device=self.global_model().device)
        return canary

    def count_canaries(self, clients, canary):
        canary_count = 0
        for client in clients:
            for batch in client.dataset._user_batches:
                for sample in batch["features"]:
                    if torch.equal(sample, canary.data[0]):
                        canary_count += 1
                        self.logger.debug(f"Client {client.name} has a canary")
                        self.logger.debug(f"Client has {len(batch['features'])} samples")
        self.logger.debug(f"Canary count {canary_count}")

    def _design_canary(self, canary_designer, mock_clients, exact_testing=False):
        # Note: Deepcopying the fl_get_module() between epochs breaks Opacus' GradSampleModule()... safer to load state_dict
        model_args = {"num_classes": self.cfg.args.num_classes, "dropout_rate": self.cfg.args.fl_dropout} | vars(self.global_model().fl_get_module())
        if self.cfg.args.dataset == "femnist":
            model_args["in_channels"] = 1
            
        design_model = self.global_model().fl_get_module().__class__(**model_args)
        design_model.load_state_dict(self.global_model().fl_get_module().state_dict()) 
        design_model =  ModuleValidator.fix_and_validate(design_model)
        canary_designer.set_grad_sample_module(GradSampleModule(copy.deepcopy(design_model)))

        # Generate canary
        canary = self._design_canary_from_mock_clients(mock_clients, canary_designer, model=design_model, exact_testing=exact_testing)
        return canary

    def _get_global_round_num(self, epoch, round, num_rounds_in_epoch):
        return (epoch - 1) * num_rounds_in_epoch + round

    def _setup_canary_designer(self, checkpoint=None):
        self.cfg.args.server_clip_const = float("inf") 
        if self.is_user_level_dp or not self.cfg.args.canary_design_reverse_server_clip or "privacy_setting" in self.server.cfg:
            self.cfg.args.server_clip_const = self.server.cfg.privacy_setting.clipping_value
        self.image_size = 32
        self.in_channels = 3
        if self.cfg.args.dataset in ["sent140", "shakespeare"]:
            canary_designer = CanaryDesignerNLP(None, logger=self.logger, local_updates=self.cfg.args.num_local_updates, local_epochs=self.cfg.args.client_epochs, optimizer_config=self.cfg.client.optimizer, client_lr=self.cfg.client.optimizer.lr,
                                                **self.cfg.args)
        else:
            if self.cfg.args.dataset == "CIFAR10":
                canary_preprocess = lambda x: normalize(x, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # noqa
            elif self.cfg.args.dataset == "femnist":
                canary_preprocess = lambda x: normalize(x, (0.1307,), (0.3081, )) # noqa
                self.image_size = 28
                self.in_channels=1
            else:
                canary_preprocess = lambda x: normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # noqa
                
            if self.cfg.args.prettify_samples:
                canary_preprocess = lambda x: x # noqa
                self.image_size = 128 if self.cfg.args.dataset == "celeba" else self.image_size
            
            canary_designer = CanaryDesigner(None, logger=self.logger, local_updates=self.cfg.args.num_local_updates, local_epochs=self.cfg.args.client_epochs, optimizer_config=self.cfg.client.optimizer, client_lr=self.cfg.client.optimizer.lr,
                                                in_channels=self.in_channels, image_size=self.image_size, canary_preprocess=canary_preprocess, **self.cfg.args)

        # Create analyser
        args = OmegaConf.to_container(copy.deepcopy(self.cfg.args))
        args.pop("plot_path")
        analyser_args = args | canary_designer.get_analyser_args() #py3.9
        analyser_args["grad_sample_module"] = None
        if checkpoint:
            analyser_args["checkpoint_train_acc"] = checkpoint["train_acc"]
            analyser_args["checkpoint_test_acc"] = checkpoint["test_acc"]

        canary_analyser = CanaryAnalyser(plot_path=self.cfg.plot_path, result_path=self.cfg.result_path, **analyser_args)

        return canary_designer, canary_analyser

    def _cleanup_post_canary_test(self, clients, target_client, replaced_data):
        if self.cfg.args.insert_canary_as_batch:
            target_client.dataset._user_batches = replaced_data
        else:
            target_client.dataset._user_batches[self.cfg.args.canary_insert_batch_index]["features"][0] = replaced_data[0]
            target_client.dataset._user_batches[self.cfg.args.canary_insert_batch_index]["labels"][0] = replaced_data[1]

        target_client.has_canary_inserted = False
        for client in clients:
            client.disable_canary_testing()

    def _initialise_privacy_budget(self, checkpoint, num_rounds_in_epoch, num_int_epochs):
        if self.cfg.args.epsilon > 0 and not self.cfg.args.fl_load_checkpoint: # For calibrating sigma to training a model for the full epochs
            steps = num_int_epochs * num_rounds_in_epoch if not checkpoint else checkpoint["steps"]
            noise_multiplier = get_noise_multiplier(steps=steps, target_epsilon=self.cfg.args.epsilon, target_delta=self.server._privacy_engine.target_delta, sample_rate=self.server._privacy_engine.user_sampling_rate)
            self.logger.info(f"Noise multiplier updated={noise_multiplier}")
            self.server._privacy_engine.noise_multiplier = noise_multiplier
            self.cfg.args.sigma = noise_multiplier
        else:
            self.server._privacy_engine.noise_multiplier = self.cfg.args.sigma # Otherwise use noise multiplier

        # Restore privacy parameters from checkpoint
        if checkpoint:
            self.server._privacy_engine.target_delta = checkpoint["delta"]
            self.server._privacy_engine.noise_multiplier = checkpoint["noise_multiplier"]
            self.server._privacy_engine.steps = checkpoint["steps"] if checkpoint["steps"] != -1 else (checkpoint["epoch"]-1) * num_rounds_in_epoch + (checkpoint["round"]-1)
            priv_budget = self.server._privacy_engine.get_privacy_spent() # Checkpointed priv budget
            self.initial_privacy_budget = {"epsilon": priv_budget.epsilon, "delta": priv_budget.delta, "sigma": self.server._privacy_engine.noise_multiplier}
            self.logger.info(f"Initial priv budget {self.initial_privacy_budget}, sigma arg={self.cfg.args.sigma}")

        if self.cfg.args.override_noise_multiplier: # If we are overriding the noise multiplier
            self.server._privacy_engine.noise_multiplier = self.cfg.args.sigma
        priv_budget = self.server._privacy_engine.get_privacy_spent() # Update priv budget
        self.server.privacy_budget = priv_budget

        sampling_rate = self.server._privacy_engine.user_sampling_rate
        self.server._privacy_engine.user_sampling_rate = 1
        self.server._privacy_engine.steps = 1
        one_step_budget = self.server._privacy_engine.get_privacy_spent()
        self.one_step_budget = {"epsilon": one_step_budget.epsilon, "delta": one_step_budget.delta, "sigma": self.server._privacy_engine.noise_multiplier} # Update priv budget
        self.server._privacy_engine.user_sampling_rate = sampling_rate
        self.server._privacy_engine.steps = 0
        self.canary_analyser.sample_rate = self.server._privacy_engine.user_sampling_rate # Set sample rate in analyser for checkpointing

    def train(
        self,
        data_provider: IFLDataProvider,
        metrics_reporter: IFLMetricsReporter,
        num_total_users: int,
        distributed_world_size: int,
        rank: int = 0,
        checkpoint=None,
    ) -> Tuple[IFLModel, Any]:
        """Trains and evaluates the model, modifying the model state. Iterates over the
        number of epochs specified in the config, and for each epoch iterates over the
        number of rounds per epoch, i.e. the number of total users divided by the number
        of users per round. For each round:

            1. Trains the model in a federated way: different local models are trained
                with local data from different clients, and are averaged into a new
                global model at the end of each round.
            2. Evaluates the new global model using evaluation data, if desired.
            3. Calculates metrics based on evaluation results and selects the best model.

        Args:
            data_provider: provides training, evaluation, and test data iterables and
                gets a user's data based on user ID
            metrics_reporter: computes and reports metrics of interest such as accuracy
                or perplexity
            num_total_users: number of total users for training

        Returns:
            model, best_metric: the trained model together with the best metric

        Note:
            Depending on the chosen active user selector, we may not iterate over
            all users in a given epoch.
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Client LR {self.cfg.client.optimizer.lr}, Server LR {self.cfg.server.server_optimizer.lr}")

        # set up synchronization utilities for distributed training
        FLDistributedUtils.setup_distributed_training(
            distributed_world_size, use_cuda=self.cuda_enabled
        )  # TODO do not call distributed utils here, this is upstream responsibility
        self.logger.info(f" dist world size = {distributed_world_size}")

        if rank != 0:
            FLDistributedUtils.suppress_output()

        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        assert self.cfg.users_per_round % distributed_world_size == 0

        best_metric = None
        best_model_state = self.global_model().fl_get_module().state_dict()
        users_per_round = min(self.cfg.users_per_round, num_total_users)

        self.data_provider = data_provider
        num_rounds_in_epoch = self.rounds_in_one_epoch(num_total_users, users_per_round)
        num_users_on_worker = data_provider.num_train_users()
        self.logger.info(
            f"num_users_on_worker: {num_users_on_worker}, "
            f"users_per_round: {users_per_round}, "
            f"num_total_users: {num_total_users}"
        )
        # torch.multinomial requires int instead of float, cast it as int
        users_per_round_on_worker = int(users_per_round / distributed_world_size)
        self.logger.debug("Validating users per round...")
        self._validate_users_per_round(users_per_round_on_worker, num_users_on_worker)
        self.logger.debug("Users validated...")

        mock_client_indices, batches_per_client, samples_per_batch = self._select_mock_client_indices(num_users_on_worker, users_per_round)

        mock_clients = self._select_mock_clients(mock_client_indices)
        self.logger.debug(f"Mock clients selected {[client.name for client in mock_clients]}")

        canary_designer, canary_analyser = self._setup_canary_designer(checkpoint=checkpoint)

        self.canary_analyser = canary_analyser
        # main training loop
        num_int_epochs = math.ceil(self.cfg.epochs)
        # starting_epoch = checkpoint["epoch"] if checkpoint else 1
        # self.logger.info(f"Starting from epoch={starting_epoch}")

        starting_epoch = 1
        for epoch in tqdm(
            range(starting_epoch, num_int_epochs + 1), desc="Epoch", unit="epoch", position=0
        ):
            for r in tqdm(
                range(1, num_rounds_in_epoch + 1),
                desc="Round",
                unit="round",
                position=0,
            ):
                display_gpu_mem(self.global_model().device, prefix=f"FLSim e={epoch}, r={r}", logger=self.logger)

                timeline = Timeline(
                    epoch=epoch, round=r, rounds_per_epoch=num_rounds_in_epoch
                )

                t = time()
                
                freeze_model = True
                while freeze_model and not self.stop_fl_training(epoch=epoch, round=r, num_rounds_in_epoch=num_rounds_in_epoch):
                    freeze_model = self._is_freeze_round(epoch, r, num_rounds_in_epoch)

                    clients = self._client_selection(
                        num_users_on_worker,
                        users_per_round_on_worker,
                        data_provider,
                        self.global_model(),
                        epoch,
                        mock_client_indices
                    )

                    if epoch == starting_epoch and r==1 and self.canary_analyser.num_tests == 0: # Initialise once
                        self._initialise_privacy_budget(checkpoint, num_rounds_in_epoch, num_int_epochs)

                    display_gpu_mem(device=self.global_model().device, prefix=f"After client selection e={epoch}, r={r}", logger=self.logger)
                    self.logger.info(f"\n ====== Epoch={epoch}, Round r={r}, model frozen={freeze_model}, frozen round={self.canary_analyser.num_tests+1} ====== ")
                    self.logger.info(f"Client Selection took: {time() - t} s.")
                    self.logger.debug(f"Selected clients: {[client.name for client in clients]}")
                    self.logger.debug(f"Model Hash: {sum([p.flatten().sum() for p in self.global_model().fl_get_module().parameters()]).item()}")
                    self.logger.info(f"Total number of canary tests - {self.canary_analyser.num_tests}")
                    self._output_design_rounds(epoch, r, num_rounds_in_epoch)

                    # Check whether to end training and checkpoint model
                    if self.cfg.checkpoint_only and self._is_canary_design_round(epoch=epoch, round=r, num_rounds_in_epoch=num_rounds_in_epoch):
                        self._checkpoint_model(epoch, r, final=True, mock_client_indices=mock_client_indices)
                        self.logger.info("Model has reached final checkpoint, checkpoint saved, exiting FLSim...")
                        return self.global_model(), best_metric

                    # Canary design round
                    if self._is_canary_design_round(epoch, r, num_rounds_in_epoch): # Generate canary and insert into client
                        self.logger.info(f"\n ===== Designing canary at epoch={epoch}, round={r} =====")
                        exact_testing = False
                        if self.cfg.args.canary_setup == "exact": # For debugging purposes
                            mock_clients = clients[1:]
                            exact_testing = True

                        canary = self._design_canary(canary_designer, mock_clients, exact_testing=exact_testing)
                        self.canary_analyser.set_canary(canary)
                        
                        # TODO: Rework this
                        self.canary_analyser.canary_losses = canary_designer.canary_losses
                        self.canary_analyser.canary_norms = canary_designer.canary_norms
                        self.canary_analyser.canary_design_bias = canary_designer.canary_design_bias
                        self.canary_analyser.benchmark_times = canary_designer.benchmark_times
                        self.canary_analyser.actual_minibatch_size = canary_designer.canary_design_minibatch_size
                        self.canary_analyser.actual_pool_size = canary_designer.canary_design_pool_size
                        self.canary_analyser.actual_sample_size = canary_designer.canary_design_sample_size
                        
                    if self._is_canary_insert_round(epoch, r, num_rounds_in_epoch):
                        type = "init" if self._is_init_canary_insert_round(epoch, r, num_rounds_in_epoch) else "canary"
                        self.logger.debug(f"Inserting canary client, epoch={epoch}, round r={r}, type={type}")
                        self.logger.info(f"Clients {[client.name for client in clients]}")

                        target_client, replaced_data = self._push_canary_to_client(clients, canary, target_batch_idx=self.cfg.args.canary_insert_batch_index, insert_as_batch=self.cfg.args.insert_canary_as_batch, type=type)
                        self.count_canaries(clients, canary)
                        
                    agg_metric_clients = self._choose_clients_for_post_aggregation_metrics(
                        train_clients=clients,
                        num_total_users=num_users_on_worker,
                        users_per_round=users_per_round_on_worker,
                    )

                    # training on selected clients for the round
                    self.logger.info(f"# clients/round on worker {rank}: {len(clients)}.")
                    aggregated_model = self._train_one_round(
                        timeline=timeline,
                        clients=clients,
                        agg_metric_clients=agg_metric_clients,
                        users_per_round=users_per_round,
                        metrics_reporter=metrics_reporter
                        if self.cfg.report_train_metrics
                        else None,
                        freeze_model=freeze_model,
                    )
                                       
                    # Testing for canary 
                    if self._is_canary_test_round(epoch, r, num_rounds_in_epoch):
                        type = "with" if (self.canary_analyser.num_tests % 2 == 0) else "without"
                        type = "with_init" if self._is_init_canary_insert_round(epoch, r, num_rounds_in_epoch) else type
                        self.logger.info(f"==== Testing aggregated server update for canary presence... type={type} ====")

                        num_clients = len(clients)
                        self.canary_analyser.test_against_agg_grad(canary, aggregated_model, self.cfg.client.optimizer.lr, num_clients, clip_factor=1, type=type)
                        self.logger.debug(f"Current dot prods {self.canary_analyser.canary_dot_prods, len(self.canary_analyser.canary_dot_prods['with_canary']), len(self.canary_analyser.canary_dot_prods['without_canary'])}")
                        
                        if hasattr(self.server, "_clip_factors"):
                            self.canary_analyser.add_clip_rate((len(self.server._clip_factors)-(np.array(self.server._clip_factors) == 1).sum()) / len(clients))
                            self.logger.info(f"Clip rate of client updates: {self.canary_analyser.clip_rates[-1]}")

                    if hasattr(self.server, "_clip_factors"):
                        self.server._clip_factors = [] # Reset 
                        
                    # report training success rate and training time variance
                    if rank == 0:
                        if (
                            self._timeout_simulator.sample_mean_per_user != 0
                            or self._timeout_simulator.sample_var_per_user != 0
                        ):
                            self.logger.info(
                                f"mean training time/user: "
                                f"{self._timeout_simulator.sample_mean_per_user}",
                                f"variance of training time/user: "
                                f"{self._timeout_simulator.sample_var_per_user}",
                            )

                        t = time()
                        (eval_metric, best_metric, best_model_state,) = self._maybe_run_evaluation(
                            timeline=timeline,
                            data_provider=data_provider,
                            metrics_reporter=metrics_reporter,
                            best_metric=best_metric,
                            best_model_state=best_model_state,
                        )
                        if eval_metric:
                            self.accuracy_metrics["eval"].append(eval_metric["Accuracy"])
                        self.logger.info(f"Evaluation took {time() - t} s. \n")

                    if self._analyse_attack(epoch, r, num_rounds_in_epoch):
                        if not self.cfg.args.skip_acc:
                            test_metrics = self.test(data_provider=data_provider, metrics_reporter=metrics_reporter)
                            self.accuracy_metrics["test"].append(test_metrics["Accuracy"])
                        self.logger.info("Analysing canary tests...")
                        self.canary_analyser.set_accuracy_metrics(self.accuracy_metrics)

                        disable_metrics = (self.cfg.args.canary_test_type == "continuous" or self.cfg.args.canary_test_type == "train_and_freeze")
                        priv_budget = {"epsilon": self.server.privacy_budget.epsilon, "delta": self.server.privacy_budget.delta, "sigma": self.server._privacy_engine.noise_multiplier}
                        self.logger.info(f"Final priv budget {priv_budget}")
                        self.canary_analyser.analyse(initial_privacy_budget=self.initial_privacy_budget, final_privacy_budget=priv_budget, one_step_budget=self.one_step_budget, global_round=self._get_global_round_num(epoch, r, num_rounds_in_epoch), disable_bias_metrics=disable_metrics, disable_init_metrics=(disable_metrics or self.cfg.args.canary_setup == "exact"), plot_losses=(not disable_metrics), plot_hists=(not disable_metrics))
                        freeze_model = False
                        
                    self._extend_canary_attack(epoch, r, num_rounds_in_epoch) # Whether or not to extend the canary attack (i.e doing train_and_freeze or continuous attack)
                    if self.cfg.args.canary_test_type == "train_and_freeze" and self._analyse_attack(epoch, r, num_rounds_in_epoch):
                        self.canary_analyser.reset() # Reset analyser between attack periods
                        
                    if self._is_canary_insert_round(epoch, r, num_rounds_in_epoch):
                        self._cleanup_post_canary_test(clients, target_client, replaced_data) # Removes inserted canary from target client + disables canary testing flag

                if self.stop_fl_training(epoch=epoch, round=r, num_rounds_in_epoch=num_rounds_in_epoch):
                    break

            self._report_post_epoch_client_metrics(timeline, metrics_reporter)
            if self.stop_fl_training(epoch=epoch, round=r, num_rounds_in_epoch=num_rounds_in_epoch):
                break

        if rank == 0 and best_metric is not None:
            self._save_model_and_metrics(self.global_model(), best_model_state)

        return self.global_model(), best_metric
    
    def _output_design_rounds(self, epoch, round, num_rounds_in_epoch):
        current_round = self._get_global_round_num(epoch, round, num_rounds_in_epoch)
        self.logger.info(f"Current global round {current_round}")
        self.logger.info(f"Design round {self._get_design_round(current_round, num_rounds_in_epoch)}")
        self.logger.debug(f"Is canary design round? {self._is_canary_design_round(epoch, round, num_rounds_in_epoch)}")
        self.logger.debug(f"Is canary insert round? {self._is_canary_insert_round(epoch, round, num_rounds_in_epoch)}")
        self.logger.debug(f"Is canary insert init round? {self._is_init_canary_insert_round(epoch, round, num_rounds_in_epoch)}")
        self.logger.debug(f"Is canary test round? {self._is_canary_test_round(epoch, round, num_rounds_in_epoch)}")
        self.logger.debug(f"Stop testing ? {self._stop_canary_testing(epoch, round, num_rounds_in_epoch)}")

    # Canary conditions
    def _is_freeze_round(self, epoch, current_round, num_rounds_in_epoch):
        return self._is_canary_test_round(epoch, current_round, num_rounds_in_epoch) and (not self.cfg.args.canary_test_type == "continuous") # Never freeze model when doing continuous testing

    def _extend_canary_attack(self, epoch, current_round, num_rounds_in_epoch):
        current_global_round = self._get_global_round_num(epoch, current_round, num_rounds_in_epoch)
        
        if self.cfg.args.canary_insert_global_round != -1 and self.cfg.args.canary_test_type == "train_and_freeze" and self.canary_analyser.num_tests >= self.cfg.args.canary_num_test_batches:
            self.cfg.args.canary_insert_global_round = current_global_round + self.cfg.args.canary_insert_offset
        
        if self.cfg.args.canary_test_type == "continuous" and current_global_round+1 > self.cfg.args.canary_insert_global_round:
            self.cfg.args.canary_insert_global_round = current_global_round + self.cfg.args.canary_insert_offset

    def _get_design_round(self, current_global_round, num_rounds_in_epoch):
        if self.cfg.args.canary_insert_epsilon != -1 and current_global_round > 1: # Second condition is needed as server.privacy_budget.epsilon is inf to begin with
            raise NotImplementedError("Canary insert epsilon not supported yet")
            # if self.server.privacy_budget.epsilon >= self.cfg.args.canary_insert_epsilon:
            #     self.cfg.args.canary_insert_global_round = current_global_round
            # else:
            #    self.cfg.args.canary_insert_global_round = float("inf")

        if (self.cfg.args.canary_insert_train_acc != -1 or self.cfg.args.canary_insert_test_acc != -1) and not self.insert_acc_achieved:
            acc_threshold = self.cfg.args.canary_insert_test_acc if self.cfg.args.canary_insert_test_acc != -1 else  self.cfg.args.canary_insert_train_acc
            acc_list = self.accuracy_metrics["eval"] if self.cfg.args.canary_insert_test_acc != -1 else self.accuracy_metrics["train"]
            if len(acc_list) > 0 and acc_list[-1] >= acc_threshold:
                self.cfg.args.canary_insert_global_round = current_global_round+1
                self.cfg.args.canary_insert_epoch = math.ceil((current_global_round+1)/num_rounds_in_epoch)
                self.insert_acc_achieved = True
            else:
               self.cfg.args.canary_insert_global_round = float("inf")

        if self.cfg.args.canary_insert_global_round != -1:
            return self.cfg.args.canary_insert_global_round
        else:
            return self._get_global_round_num(self.cfg.args.canary_insert_epoch, 1, num_rounds_in_epoch)

    def _is_canary_design_round(self, epoch, round, num_rounds_in_epoch):
        current_round = self._get_global_round_num(epoch, round, num_rounds_in_epoch)
        design_round = self._get_design_round(current_round, num_rounds_in_epoch)
        if self.cfg.args.canary_setup == "exact":  
            return current_round >= design_round
        else:
            return current_round == design_round and (self.canary_analyser.num_tests == 0 or self.cfg.args.canary_test_type == 'continuous')

    def _is_canary_insert_round(self, epoch, round, num_rounds_in_epoch):
        current_round = self._get_global_round_num(epoch, round, num_rounds_in_epoch)
        design_round = self._get_design_round(current_round, num_rounds_in_epoch)
        return current_round >= design_round and (self.canary_analyser.num_tests % 2) == 0

    def _is_init_canary_insert_round(self, epoch, round, num_rounds_in_epoch):
        if self.cfg.args.canary_test_type == "continuous" or self.cfg.args.canary_test_type == "train_and_freeze" or self.cfg.args.canary_setup == "exact":
            return False
        else:
            return self.canary_analyser.num_tests >= self.cfg.args.canary_num_test_batches

    def _is_canary_test_round(self, epoch, round, num_rounds_in_epoch):
        current_round = self._get_global_round_num(epoch, round, num_rounds_in_epoch)
        design_round = self._get_design_round(current_round, num_rounds_in_epoch)
        return current_round == design_round
    
    def _analyse_attack(self, epoch, round, num_rounds_in_epoch):
        return (self.cfg.args.canary_test_type == "continuous" and self.canary_analyser.num_tests >= 2 and self._is_canary_test_round(epoch, round, num_rounds_in_epoch)) \
                or (self.cfg.args.canary_test_type == "train_and_freeze" and self.canary_analyser.num_tests >= self.cfg.args.canary_num_test_batches) \
                or self._stop_canary_testing(epoch, round, num_rounds_in_epoch)

    def _stop_canary_testing(self, epoch, round, num_rounds_in_epoch):
        round_factor = 1 if self.cfg.args.canary_setup == "exact" else 1.5
        
        if self.cfg.args.canary_test_type == "continuous" or self.cfg.args.canary_test_type == "train_and_freeze": # Train until total number of epochs reached
            # return current_round >= self.cfg.args.canary_num_test_batches*self.cfg.args.canary_insert_offset # For debugging
            return False
        else: 
            return self.canary_analyser.num_tests >= self.cfg.args.canary_num_test_batches*round_factor # current round + 1 > design round

    def stop_fl_training(self, *, epoch, round, num_rounds_in_epoch) -> bool:
        """Stops FL training when the necessary number of steps/epochs have been
        completed in case of fractional epochs or if clients time out.
        """
        global_round_num = (epoch - 1) * num_rounds_in_epoch + round
        return (
            (global_round_num / num_rounds_in_epoch)
            >= self.cfg.epochs  # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            or self._timeout_simulator.stop_fl()
            or self._stop_canary_testing(epoch, round, num_rounds_in_epoch)
        )

    def _train_one_round(
        self,
        timeline: Timeline,
        clients: Iterable[Client],
        agg_metric_clients: Iterable[Client],
        users_per_round: int,
        metrics_reporter: Optional[IFLMetricsReporter],
        freeze_model: bool,
    ) -> None:
        """Trains the global model for one training round.

        Args:
            timeline: information about the round, epoch, round number, ...
            clients: clients for the round
            agg_metric_clients: clients for evaluating the post-aggregation
                training metrics
            users_per_round: the number of participating users
            metrics_reporter: the metric reporter to pass to other methods
        """
        t = time()
        self.server.init_round()
        self.logger.info(f"Round initialization took {time() - t} s.")

        def update(client):
            client_delta, weight = client.generate_local_update(
                self.global_model(), metrics_reporter
            )
            self.server.receive_update_from_client(Message(client_delta, weight), enable_clipping= not (freeze_model and self.cfg.args.canary_design_reverse_server_clip))
            if self.is_user_level_dp and hasattr(self.server, "_clip_factors"): # If user-level DP
                self.logger.debug(f"Server recieved update from {client.name}, clipping factor {(self.server._clip_factors or [None])[-1]}\n")

        t = time()
        for client in clients:
            if freeze_model: client.enable_canary_testing()
            update(client)
        self.logger.info(f"Collecting round's clients took {time() - t} s.")

        t = time()

        aggregated_model = None
        if freeze_model:
            aggregated_model = self.server._aggregator.aggregate()
            self.server._privacy_engine.steps -= 1 # Freezing the model, fake step
            self.logger.info(f"Noise sensitivity {self.server._clipping_value} / {self.server._aggregator.sum_weights.item()}")
            self.server._privacy_engine.add_noise(
                aggregated_model,
                self.server._clipping_value / self.server._aggregator.sum_weights.item(),
            )
            self.server._privacy_engine.get_privacy_spent()
        else:
           aggregated_model = self.server.step()
            
        self.logger.info(f"Finalizing round took {time() - t} s.")

        t = time()
        train_metrics = self._report_train_metrics(
            model=self.global_model(),
            timeline=timeline,
            metrics_reporter=metrics_reporter,
        )
        if train_metrics:
            self.accuracy_metrics["train"].append(train_metrics["Accuracy"])

        eval_metrics = self._evaluate_global_model_after_aggregation_on_train_clients(
            clients=agg_metric_clients,
            model=self.global_model(),
            timeline=timeline,
            users_per_round=users_per_round,
            metrics_reporter=metrics_reporter,
        )
        if eval_metrics:
            self.accuracy_metrics["agg"].append(eval_metrics["Accuracy"])
            
        self._calc_post_epoch_communication_metrics(
            timeline,
            metrics_reporter,
        )
        self.logger.info(f"Aggregate round reporting took {time() - t} s.\n")

        return aggregated_model

    def _choose_clients_for_post_aggregation_metrics(
        self,
        train_clients: Iterable[Client],
        num_total_users: int,
        users_per_round: int,
    ) -> Iterable[Client]:
        """Chooses clients for the post-aggregation training metrics.
        Depending on the config parameters, either returns the round's
        training clients or new randomly drawn clients.
        """
        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        if self.cfg.use_train_clients_for_aggregation_metrics:
            return train_clients

        # For the post-aggregation metrics, evaluate on new users
        agg_metric_client_idcs = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            replacement=False,
        ).tolist()

        agg_metric_clients = [
            self.create_or_get_client_for_data(i, self.data_provider)
            for i in agg_metric_client_idcs
        ] # noqa
        return train_clients

    def _evaluate_global_model_after_aggregation_on_train_clients(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        timeline: Timeline,
        users_per_round: int,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ):
        if (
            metrics_reporter is not None
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            and self.cfg.report_train_metrics
            and self.cfg.report_train_metrics_after_aggregation
            and timeline.tick(1.0 / self.cfg.train_metrics_reported_per_epoch)
        ):
            with torch.no_grad():
                self._cuda_state_manager.before_train_or_eval(model)
                model.fl_get_module().eval()
                for client in clients:
                    for batch in client.dataset.train_data():
                        batch_metrics = model.get_eval_metrics(batch)
                        if metrics_reporter is not None:
                            metrics_reporter.add_batch_metrics(batch_metrics)
                model.fl_get_module().train()

            privacy_metrics = self._calc_privacy_metrics(
                clients, model, metrics_reporter
            )
            overflow_metrics = self._calc_overflow_metrics(
                clients, model, users_per_round, metrics_reporter
            )

            eval_metrics, best_replaced = metrics_reporter.report_metrics(
                model=model,
                reset=True,
                stage=TrainingStage.AGGREGATION,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=privacy_metrics + overflow_metrics,
            )
            self._cuda_state_manager.after_train_or_eval(model)

            return eval_metrics

    def _report_post_epoch_client_metrics(
        self,
        timeline: Timeline,
        metrics_reporter: Optional[IFLMetricsReporter],
    ):
        if (
            metrics_reporter is not None
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            and self.cfg.report_client_metrics
            and self.cfg.report_client_metrics_after_epoch
            and (timeline.epoch % self.cfg.client_metrics_reported_per_epoch == 0)
        ):
            client_models = {
                client: client.last_updated_model for client in self.clients.values()
            }

            client_scores = self._calc_post_epoch_client_metrics(
                client_models, timeline, metrics_reporter
            )

            # Find stats over the client_metrics (mean, min, max, median, std)
            client_stats_trackers = {}
            score_names = [metric.name for metric in next(iter(client_scores))]
            for score_name in score_names:
                client_stats_trackers[score_name] = RandomVariableStatsTracker(
                    tracks_quantiles=True
                )
            for client_metric_list in client_scores:
                for client_metric in client_metric_list:
                    client_stats_trackers[client_metric.name].update(
                        client_metric.value
                    )

            reportable_client_metrics = []
            for score_name in score_names:
                for stat_name, stat_key in [
                    ("Mean", "mean_val"),
                    ("Median", "median_val"),
                    ("Upper Quartile", "upper_quartile_val"),
                    ("Lower Quartile", "lower_quartile_val"),
                    ("Min", "min_val"),
                    ("Max", "max_val"),
                    ("Standard Deviation", "standard_deviation_val"),
                    ("Num Samples", "num_samples"),
                ]:
                    score = client_stats_trackers[score_name].__getattribute__(stat_key)
                    reportable_client_metrics.append(Metric(stat_name, score))

            metrics_reporter.report_metrics(
                model=None,
                reset=True,
                stage=TrainingStage.PER_CLIENT_EVAL,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=reportable_client_metrics,
            )

@dataclass
class CanarySyncTrainerConfig(FLTrainerConfig):
    _target_: str = fullclassname(CanarySyncTrainer)
    server: SyncServerConfig = SyncServerConfig()
    users_per_round: int = 10
    # overselect users_per_round / dropout_rate users, only use first
    # users_per_round updates
    dropout_rate: float = 1.0
    report_train_metrics_after_aggregation: bool = False
    report_client_metrics_after_epoch: bool = False
    # Whether client metrics on eval data should be computed and reported.
    report_client_metrics: bool = False
    # how many times per epoch should we report client metrics
    # numbers greater than 1 help with plotting more precise training curves
    client_metrics_reported_per_epoch: int = 1
    plot_path: str = "./"
    result_path: str = "./test.tar"
    checkpoint_round: float = 0.1
    save_checkpoint: bool = False
    checkpoint_only: bool = True
    load_checkpoint: bool = True
    args: Optional[Dict[str, Any]] = None