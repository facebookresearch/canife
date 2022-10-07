#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train a binary classifier on LEAF's CelebA dataset with FLSim.

Before running this file, you need to download the dataset and partition the data by users.
1. Clone the leaf dataset by running `git clone https://github.com/TalwalkarLab/leaf.git`
2. Change direectory to celeba: `cd leaf/data/celeba || exit`
3. Download the data from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    - Download or request the metadata files `identity_CelebA.txt` and `list_attr_celeba.txt`,
      and place them inside the data/raw folder.
    - Download the celebrity faces dataset from the same site. Place the images in a folder
       named `img_align_celeba` in the same folder as above.
4. Run the pre-processing script:
    - `./preprocess.sh --sf 0.01 -s niid -t 'user' --tf 0.90 -k 1 --spltseed 1`

Typical usage example:
    python3 celeba_example.py --config-file configs/celeba_config.json
"""
import json
import os
import random
from typing import Any, Iterator, List, Tuple

import flsim.configs  # noqa
import hydra  # @manual
import torch
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataProvider,
    FLModel,
    LEAFDataLoader,
    MetricsReporter,
    Resnet18,
    SimpleConvNet,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from canife.utils import get_plot_path


class CelebaDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_root,
        num_users=None,
        transform=None,
        target_transform=None,
    ):
        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        user_ids = self.dataset["users"]
        num_users = num_users if num_users is not None else len(user_ids)
        user_ids = random.sample(user_ids, min(len(user_ids), num_users))

        self.transform = transform
        self.target_transform = target_transform

        self.image_root = image_root
        self.image_folder = ImageFolder(image_root, transform)
        self.data = {}
        self.targets = {}
        # Populate self.data and self.targets
        for user_id, user_data in self.dataset["user_data"].items():
            if user_id in user_ids:
                self.data[user_id] = [
                    int(os.path.splitext(img_path)[0]) for img_path in user_data["x"]
                ]
                self.targets[user_id] = list(user_data["y"])

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Any]]]:
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str) -> Tuple[List[torch.Tensor], List[Any]]:

        if user_id not in self.data or user_id not in self.targets:
            raise IndexError(f"User {user_id} is not in dataset")

        user_imgs = []
        for image_index in self.data[user_id]:
            user_imgs.append(self.image_folder[image_index - 1][0])
        user_targets = self.targets[user_id]

        if self.target_transform is not None:
            user_targets = [self.target_transform(target) for target in user_targets]

        return user_imgs, user_targets

    def __len__(self) -> int:
        return len(self.data)


def build_data_provider(data_config, trainer_config):
    IMAGE_SIZE: int = 32
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Local testing
    # train_split = "/data/train/all_data_0_01_keep_0_train_9.json" if "celeba_iid" not in trainer_config.args.dataset else "/data/train/all_data_0_01_01_keep_0_train_9_iid.json" 
    # test_split = "/data/test/all_data_0_01_keep_0_test_9.json"  if "celeba_iid" not in trainer_config.args.dataset  else "/data/test/all_data_0_01_01_keep_0_test_9_iid.json" 

    train_split = "/data/train/all_data_0_0_keep_0_train_9.json" if  "celeba_iid" not in trainer_config.args.dataset else "/data/train/all_data_0_0_0_keep_0_train_9_iid.json" 
    test_split = "/data/test/all_data_0_0_keep_0_test_9.json"  if  "celeba_iid" not in trainer_config.args.dataset  else "/data/test/all_data_0_0_0_keep_0_test_9_iid.json" 

    train_dataset = CelebaDataset( # data_root arg should be leaf/celeba 
        data_root=data_config.data_root + train_split,
        image_root=data_config.data_root+"/data/raw/",
        transform=transform,
    )
    test_dataset = CelebaDataset(
        data_root=data_config.data_root + test_split,
        transform=transform,
        image_root=train_dataset.image_root,
    )

    print(
        f"Created datasets with {len(train_dataset)} train users and {len(test_dataset)} test users"
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=data_config.drop_last,
    )

    # data_provider = LEAFDataProvider(dataloader)
    data_provider = DataProvider(dataloader)

    print(f"Training clients in total: {data_provider.num_train_users()}")

    return data_provider

def _get_checkpoint_path(cfg):
    filename =  cfg.args.checkpoint_path
    filename += f"/FLSim_dp={cfg.args.dp_level}_model={cfg.args.model_arch}_dataset={cfg.args.dataset}_num_clients={cfg.args.users_per_round}_test_size={cfg.args.local_batch_size}"
    filename += f"_insert_test_acc={cfg.args.canary_insert_test_acc}_insert_train_acc={cfg.args.canary_insert_train_acc}"
    filename += ".tar"
    return filename

def main_worker(
    trainer_config,
    data_config,
    model_config,
    use_cuda_if_available: bool = True,
    distributed_world_size: int = 1,
) -> None:

    checkpoint_path = _get_checkpoint_path(trainer_config)
    if (trainer_config.args.fl_load_checkpoint) and not os.path.isfile(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist, experiment exiting early...")
        return 

    data_provider = build_data_provider(data_config, trainer_config)
    
    for exp_num in range(0, data_config.canary_iters):
        torch.cuda.empty_cache()

        if not data_config.debug_config:
            trainer_config["plot_path"] = get_plot_path(trainer_config.args, exp_num=exp_num, file_suffix="")
            trainer_config["result_path"] = get_plot_path(trainer_config.args, exp_num, ".tar")

        if model_config.model_arch == "resnet":
            model = Resnet18(num_classes=2)
        else:
            model = SimpleConvNet(num_classes=2, dropout_rate=model_config.dropout)

        cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
        device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
        print(model)
        # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
        global_model = FLModel(model, device)
        if cuda_enabled:
            global_model.fl_cuda()

        trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

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


@hydra.main(config_path=None, config_name="celeba_config", version_base="1.1")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data
    model_config = cfg.model

    main_worker(
        trainer_config,
        data_config,
        model_config,
        cfg.use_cuda_if_available,
        cfg.distributed_world_size,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
