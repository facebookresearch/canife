#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.channels.message import Message
from flsim.channels.pq_utils.pq import PQ
from flsim.channels.product_quantization_channel import ProductQuantizationChannel
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import FedAvgOptimizerConfig
from flsim.privacy.common import PrivacyBudget, PrivacySetting
from flsim.privacy.privacy_engine import IPrivacyEngine
from flsim.privacy.privacy_engine_factory import NoiseType, PrivacyEngineFactory
from flsim.privacy.user_update_clip import UserUpdateClipper
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import ISyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class SyncDPSGDServer(ISyncServer):
    """
    User level DP-SGD Server implementing https://arxiv.org/abs/1710.06963

    Args:
        global_model: IFLModel: Global (server model) to be updated between rounds
        users_per_round: int: User per round to calculate sampling rate
        num_total_users: int: Total users in the dataset to calculate sampling rate
        Sampling rate = users_per_round / num_total_users
        channel: Optional[IFLChannel]: Communication channel between server and clients
    """

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncDPSGDServerConfig,
            **kwargs,
        )
        assert (
            self.cfg.aggregation_type == AggregationType.AVERAGE  # pyre-ignore[16]
        ), "DP training must be done with simple averaging and uniform weights."

        self.privacy_budget = PrivacyBudget()
        self._clipping_value = self.cfg.privacy_setting.clipping_value
        self._optimizer = instantiate(
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )
        self._global_model: IFLModel = global_model
        self._user_update_clipper: UserUpdateClipper = UserUpdateClipper()
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._privacy_engine: Optional[IPrivacyEngine] = None
        self._active_user_selector = instantiate(self.cfg.active_user_selector)
        self._channel: IFLChannel = channel or IdentityChannel()

        self._clip_factors = [] # TODO: Canary modification
    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
        if OmegaConf.is_missing(cfg.server_optimizer, "_target_"):
            cfg.server_optimizer = FedAvgOptimizerConfig()

    @property
    def global_model(self):
        return self._global_model

    def select_clients_for_training(
        self,
        num_total_users,
        users_per_round,
        data_provider: Optional[IFLDataProvider] = None,
        epoch: Optional[int] = None,
    ):
        if self._privacy_engine is None:
            self._privacy_engine: IPrivacyEngine = PrivacyEngineFactory.create(
                # pyre-ignore[16]
                self.cfg.privacy_setting,
                users_per_round,
                num_total_users,
                noise_type=NoiseType.GAUSSIAN,
            )
            self._privacy_engine.attach(self._global_model.fl_get_module())
        return self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_model=self.global_model,
            epoch=epoch,
        )

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()
        self._privacy_engine.attach(self._global_model.fl_get_module())


    def _init_and_boradcast_qparams(self, client_delta):
        if isinstance(self._channel, ProductQuantizationChannel):
            seed_centroids = {}
            state_dict = client_delta.fl_get_module().state_dict()
            for name, param in state_dict.items():
                if (
                    param.ndim > 1
                    and param.numel() >= self._channel.cfg.min_numel_to_quantize
                ):
                    pq = PQ(
                        param.data.size(),
                        self._channel.cfg.max_block_size,
                        self._channel.cfg.num_codebooks,
                        self._channel.cfg.max_num_centroids,
                        self._channel.cfg.num_k_means_iter,
                        self._channel.cfg.verbose,
                    )
                    centroids, _ = pq.encode(param.data.cpu())
                    seed_centroids[name] = centroids
            self._channel.seed_centroids = seed_centroids

    # TODO: Canary modification, for debugging
    def _output_model(self, model, message="", scale=False):
        if scale:
            print(f"{message} {torch.cat([p.detach().flatten()*(1.0/0.8991111469452855)*4 for p in model.parameters()])}")
        else:
            print(f"{message} {torch.cat([p.flatten() for p in model.parameters()])}")

    def receive_update_from_client(self, message: Message, enable_clipping=True):
        message = self._channel.client_to_server(message)
        # self._output_model(message.model.fl_get_module(), "Model update receieved")
        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )
        # self._output_model(message.model.fl_get_module(), "Model update applied weight")

        # TODO: Needed to scale canary grads
        clip_factor = self._user_update_clipper.clip(
            message.model.fl_get_module(), max_norm=self._clipping_value, enable_clipping=enable_clipping
        )
        # self._output_model(message.model.fl_get_module(), "Model update clipped")
        # self._output_model(message.model.fl_get_module(), "Model update clipped and rescaled", scale=True)
        self._clip_factors.append(clip_factor) # TODO: Canary modification

        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

    def step(self):
        assert self._privacy_engine is not None, "PrivacyEngine is not initialized"

        aggregated_model = self._aggregator.aggregate(distributed_op=OperationType.SUM)

        if FLDistributedUtils.is_master_worker():
            self._privacy_engine.add_noise(
                aggregated_model,
                self._clipping_value / self._aggregator.sum_weights.item(),
            )

        FLDistributedUtils.synchronize_model_across_workers(
            operation=OperationType.BROADCAST,
            model=aggregated_model,
            weights=self._aggregator.sum_weights,
        )

        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )
        self._optimizer.step()
        self.privacy_budget = self._privacy_engine.get_privacy_spent()

        return aggregated_model # TODO: Canary modification.

@dataclass
class SyncDPSGDServerConfig(SyncServerConfig):
    _target_: str = fullclassname(SyncDPSGDServer)
    aggregation_type: AggregationType = AggregationType.AVERAGE
    privacy_setting: PrivacySetting = PrivacySetting()
