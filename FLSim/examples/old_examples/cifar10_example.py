#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train an image classifier with FLSim to simulate a federated learning training environment.

With this tutorial, you will learn the following key components of FLSim:
1. Data loading
2. Model construction
3. Trainer construction

    Typical usage example:
    python3 cifar10_example.py --config-file configs/cifar10_config.json
"""

import flsim.configs  # noqa
import hydra
import torch
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    FLModel,
    MetricsReporter,
    Resnet18,
    SimpleConvNet,
)
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from canife.utils import get_plot_path


IMAGE_SIZE = 32


def build_data_provider(data_root, local_batch_size, examples_per_user, drop_last: bool = False):

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root=data_root, train=True, download=False, transform=transform
    )
    val_dataset = CIFAR10(
        root=data_root, train=False, download=False, transform=transform
    )
    test_dataset = CIFAR10(
        root=data_root, train=False, download=False, transform=transform
    )
    sharder = SequentialSharder(examples_per_shard=examples_per_user)

    fl_data_loader = DataLoader(
        train_dataset, val_dataset, test_dataset, sharder, local_batch_size, drop_last
    )

    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider


def main(
    trainer_config,
    data_config,
    model_config,
    use_cuda_if_available: bool = True,
) -> None:
    data_provider = build_data_provider(
        data_root=data_config.data_root,
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        drop_last=False,
    )

    for exp_num in range(0, data_config.canary_iters):
        if not data_config.debug_config:
            trainer_config["plot_path"] = get_plot_path(trainer_config.args, exp_num=exp_num, file_suffix="")
            trainer_config["result_path"] = get_plot_path(trainer_config.args, exp_num, ".tar")

        cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
        device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

        if model_config.model_arch == "resnet":
            model = Resnet18(num_classes=10)
        else:
            model = SimpleConvNet(num_classes=10, dropout_rate=model_config.dropout)

        # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
        global_model = FLModel(model, device)
        if cuda_enabled:
            global_model.fl_cuda()
            
        trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
        print(f"Created {trainer_config._target_}")

        metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

        final_model, eval_score = trainer.train(
            data_provider=data_provider,
            metrics_reporter=metrics_reporter,
            num_total_users=data_provider.num_train_users(),
            distributed_world_size=1,
        )

        test_metrics = trainer.test(
            data_provider=data_provider,
            metrics_reporter=MetricsReporter([Channel.STDOUT]),
        )

        if hasattr(trainer, "canary_analyser") and trainer.canary_analyser:
            trainer.accuracy_metrics["test"].append(test_metrics["Accuracy"])
            trainer.canary_analyser.set_accuracy_metrics(trainer.accuracy_metrics)
            trainer.logger.info(f"Final accuracy metrics {trainer.accuracy_metrics}")
            trainer.logger.info("Analysing canary tests...")
            trainer.canary_analyser.analyse()
        else:
            if data_config.canary_iters > 1:
                trainer.logger.info("Experiment ended early - either checkpoint only or model failed to reach insertion epoch/accuracy for canary testing")
            return 

@hydra.main(config_path=None, config_name="cifar10_tutorial", version_base="1.1")
def run(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data
    model_config = cfg.model 

    main(
        trainer_config,
        data_config,
        model_config
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
