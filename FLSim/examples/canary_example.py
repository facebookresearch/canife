#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
"""
import copy
import json
import os
import random
from typing import Any, Iterator, List, Tuple

import flsim.configs  # noqa
import hydra  # @manual
import numpy as np
import torch
import torch.nn as nn
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    FLModel,
    LEAFDataLoader,
    LEAFDataProvider,
    MetricsReporter,
    Resnet18,
    SequentialSharder,
    SimpleConvNet,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.cifar import CIFAR10

from canife.utils import TextProcessorSent140, TextProcessorShakes, get_plot_path


IMAGE_SIZE = 32

# Datasets 
class ShakespeareDataset(Dataset):
    SEED = 7

    def __init__(
        self,
        data_root=None,
        num_users=None,
    ):
        self.text_processor = TextProcessorShakes()
        with open(data_root, "r") as f:
            dataset = json.load(f)

        user_ids = dataset["users"]
        random.seed(self.SEED)
        num_users = num_users if num_users is not None else len(user_ids)
        user_ids = random.sample(user_ids, min(len(user_ids), num_users))
        print(f"Creating dataset with {num_users} users")

        # Filter train and test datasets based on user_ids list
        self.dataset = dataset
        self.data = {}
        self.targets = {}
                
        # Populate self.data and self.targets
        for user_id, user_data in self.dataset["user_data"].items():
            if user_id not in user_ids:
                continue
            self.data[user_id] = list(user_data["x"])
            self.targets[user_id] = list(user_data["y"])

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Any]]]:
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str) -> Tuple[List[torch.Tensor], List[Any]]:

        if user_id not in self.data or user_id not in self.targets:
            raise IndexError(f"User {user_id} is not in dataset")

        user_utterances = self.process_x(self.data[user_id])
        user_targets = self.process_y(self.targets[user_id])
        return user_utterances, user_targets

    def __len__(self) -> int:
        return len(self.data)

    def get_user_ids(self):
        return self.data.keys()

    def process_x(self, raw_x_batch):
        x_batch = [self.text_processor.word_to_indices(word) for word in raw_x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [self.text_processor.letter_to_vec(c) for c in raw_y_batch]
        return y_batch

class CelebaDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_root,
        num_users=None,
        transform=None,
        target_transform=None,
    ):
        with open(data_root, "r") as f:
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

class Sent140Dataset(Dataset):
    def __init__(self, data_root, max_seq_len):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.text_processor = TextProcessorSent140()
        self.vocab_size = self.text_processor.vocab_size
        self.embedding_size = 300

        with open(data_root, "r") as f:
            self.dataset = json.load(f)

        self.data = {}
        self.targets = {}

        self.num_classes = 2

        # Populate self.data and self.targets
        for user_id, user_data in self.dataset["user_data"].items():
            self.data[user_id] = self.process_x(list(user_data["x"]))
            self.targets[user_id] = self.process_y(list(user_data["y"]))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str):
        if user_id not in self.data or user_id not in self.targets:
            raise IndexError(f"User {user_id} is not in dataset")

        return self.data[user_id], self.targets[user_id]

    def process_x(self, raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [self.text_processor.line_to_indices(e, self.max_seq_len) for e in x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        return y_batch

class FemnistDatasetChunked(Dataset):
    IMAGE_SIZE = (28, 28)
    def __init__(
        self,
        data_root,
        num_users=None,
        transform=None,
        target_transform=None,
    ):
        with open(data_root, "r") as f:
            dataset = json.load(f)

        user_ids = []
        for _, chunk_data in dataset:
            user_ids.extend(list(chunk_data["user_data"].keys()))

        num_users = num_users if num_users is not None else len(user_ids)
        user_ids = random.sample(user_ids, min(len(user_ids), num_users))
        print(f"Creating dataset with {num_users} users")

        self.transform = transform
        self.transform = transform
        self.target_transform = target_transform

        self.data = {}
        self.targets = {}
        # Populate self.data and self.targets
        for _, chunk_data in dataset:
            for user_id in user_ids:
                if user_id in set(chunk_data["users"]):
                    self.data[user_id] = [
                        np.array(img) for img in chunk_data["user_data"][user_id]["x"]
                    ]
                    self.targets[user_id] = list(chunk_data["user_data"][user_id]["y"])

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Any]]]:
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str) -> Tuple[List[torch.Tensor], List[Any]]:

        if user_id not in self.data or user_id not in self.targets:
            return [], []

        user_imgs, user_targets = self.data[user_id], self.targets[user_id]
        user_imgs = [
            Image.fromarray(img.reshape(FemnistDataset.IMAGE_SIZE)) for img in user_imgs
        ]

        user_imgs = [self.transform(img) for img in user_imgs]

        if self.target_transform is not None:
            user_targets = [self.target_transform(target) for target in user_targets]

        return user_imgs, user_targets

    def __len__(self) -> int:
        return len(self.data)

class FemnistDataset(Dataset):
    IMAGE_SIZE = (28, 28)
    def __init__(
        self,
        data_root,
        num_users=None,
        transform=None,
        target_transform=None,
    ):
        with open(data_root, "r") as f:
            dataset = json.load(f)

        user_ids = dataset["users"]
        num_users = num_users if num_users is not None else len(user_ids)
        user_ids = random.sample(user_ids, min(len(user_ids), num_users))
        print(f"Creating dataset with {num_users} users")

        self.transform = transform
        self.transform = transform
        self.target_transform = target_transform

        self.data = {}
        self.targets = {}
        # Populate self.data and self.targets
        for user_id in user_ids:
            if user_id in set(dataset["users"]):
                self.data[user_id] = [
                    np.array(img) for img in dataset["user_data"][user_id]["x"]
                ]
                self.targets[user_id] = list(dataset["user_data"][user_id]["y"])

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Any]]]:
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str) -> Tuple[List[torch.Tensor], List[Any]]:

        if user_id not in self.data or user_id not in self.targets:
            return [], []

        user_imgs, user_targets = self.data[user_id], self.targets[user_id]
        user_imgs = [
            Image.fromarray(img.reshape(FemnistDataset.IMAGE_SIZE)) for img in user_imgs
        ]

        user_imgs = [self.transform(img) for img in user_imgs]

        if self.target_transform is not None:
            user_targets = [self.target_transform(target) for target in user_targets]

        return user_imgs, user_targets

    def __len__(self) -> int:
        return len(self.data)

# NLP Models
class Sent140StackedLSTMModel(nn.Module):
    def __init__(
        self, seq_len, num_classes, emb_size, n_hidden, vocab_size, dropout_rate, **kwargs
    ):
        super(Sent140StackedLSTMModel, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.stacked_lstm = nn.LSTM(
            self.emb_size, self.n_hidden, 2, batch_first=True, dropout=self.dropout_rate
        )
        self.fc1 = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        # self.out = nn.Linear(128, self.num_classes)

    def set_embedding_weights(self, emb_matrix, trainable=False):
        self.embedding.weight = torch.nn.Parameter(emb_matrix)
        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, features, input_embeds=None):
        # seq_lens = torch.sum(features != (self.vocab_size - 1), 1) - 1
        if features is not None:
            x = self.embedding(features)
        else:
            x = input_embeds

        outputs, _ = self.stacked_lstm(x)
        # outputs = outputs[torch.arange(outputs.size(0)), seq_lens]

        pred = self.fc1(self.dropout(outputs[:, -1]))
        return pred

class ShakespeareModel(nn.Module):
    def __init__(self, seq_len, num_classes, n_hidden, dropout_rate=0.0, **kwargs):
        super(ShakespeareModel, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes  # Number of characters supported
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(self.num_classes, 8)
        self.stacked_lstm = nn.LSTM(
            8, self.n_hidden, 2, batch_first=True, dropout=self.dropout_rate
        )
        self.out = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, features, input_embeds=None):
        if features is not None:
            x = self.embedding(features)
        else:
            x = input_embeds
        outputs, _ = self.stacked_lstm(x)
        pred = self.out(outputs[:, -1])
        return pred

# Data providers

def build_data_provider_shakespeare(data_config):
    # Local testing
    # train_split = "/data/train/all_data_0_2_keep_0_train_9.json"
    # test_split = "/data/test/all_data_0_2_keep_0_test_9.json"
    
    # Full splits
    train_split = "/data/train/all_data_0_0_keep_0_train_9.json"
    test_split = "/data/test/all_data_0_0_keep_0_test_9.json"
    
    train_dataset = ShakespeareDataset(data_root=data_config.data_root + train_split)
    test_dataset = ShakespeareDataset(data_root=data_config.data_root + test_split)

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=True,
    )

    data_provider = LEAFDataProvider(dataloader)
    return data_provider
    
def build_data_provider_sent140(
    local_batch_size, vocab_size, num_users, user_dist, max_seq_len, drop_last, data_path
):
    train_dataset = Sent140Dataset(
        data_root=data_path + "/data/train/all_data_0_15_keep_1_train_6.json",
        max_seq_len=max_seq_len,
    )
    eval_dataset = Sent140Dataset(
        data_root=data_path + "/data/test/all_data_0_15_keep_1_test_6.json",
        max_seq_len=max_seq_len,
    )
    test_dataset = Sent140Dataset(
        data_root=data_path + "/data/test/all_data_0_15_keep_1_test_6.json",
        max_seq_len=max_seq_len,
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        eval_dataset,
        test_dataset,
        batch_size=local_batch_size,
        drop_last=drop_last,
    )

    data_provider = LEAFDataProvider(dataloader)
    return data_provider, train_dataset.vocab_size, train_dataset.embedding_size

def build_data_provider_cifar10(data_root, local_batch_size, examples_per_user, drop_last: bool = False, disable_aug=False):
    
    if disable_aug:
        transform_list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    else:
        transform_list = [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    
    transform = transforms.Compose(transform_list)
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

def build_data_provider_celeba(data_config, trainer_config, disable_aug=False):
    IMAGE_SIZE: int = 32
    if disable_aug:
        IMAGE_SIZE = 128
        transform_list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    else:
        transform_list = [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
    transform = transforms.Compose(transform_list)

    # Local testing
    # train_split = "/data/train/all_data_0_01_keep_0_train_9.json" if not "celeba_iid" in trainer_config.args.dataset else "/data/train/all_data_0_01_01_keep_1_train_9.json" 
    # test_split = "/data/test/all_data_0_01_keep_0_test_9.json"  if not "celeba_iid" in trainer_config.args.dataset  else "/data/test/all_data_0_01_01_keep_1_test_9.json" 

    # GPU Debug (Non-IID)
    # train_split = "/data/train/all_data_0_1_keep_1_train_9.json"  
    # test_split = "/data/test/all_data_0_1_keep_1_test_9.json"

    train_split = "/data/train/all_data_0_0_keep_0_train_9.json" if "celeba_iid" not in trainer_config.args.dataset else "/data/train/all_data_0_0_0_keep_0_train_9_iid.json" 
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

def build_data_provider_femnist(data_config, disable_aug=False):
    if disable_aug:
        transform_list = [transforms.ToTensor()]
    else:
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),]
        
    transform = transforms.Compose(transform_list)
    
    # Local debugging
    train_split = data_config.data_root + "/data/train/all_data_0_niid_05_keep_0_train_9.json"
    test_split = data_config.data_root + "/data/test/all_data_0_niid_05_keep_0_test_9.json" 

    train_dataset = FemnistDataset(
        data_root=train_split,
        transform=transform,
    )
    test_dataset = FemnistDataset(
        data_root=test_split,
        transform=transform,
    )

    print(
        f"Created datasets with {len(train_dataset)} train users and {len(test_dataset)} test users"
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
    )
    data_provider = LEAFDataProvider(dataloader)
    print(f"Training clients in total: {data_provider.num_train_users()}")
    return data_provider
    
def _get_checkpoint_path(cfg):
    filename =  cfg.args.checkpoint_path
    filename += f"/FLSim_dp={cfg.args.dp_level}_model={cfg.args.model_arch}_dataset={cfg.args.dataset}_num_clients={cfg.args.users_per_round}_test_size={cfg.args.local_batch_size}"
    filename += f"_insert_test_acc={cfg.args.canary_insert_test_acc}_insert_train_acc={cfg.args.canary_insert_train_acc}_client_epochs={cfg.args.client_epochs}"
    if cfg.args.epsilon != -1 or cfg.args.sigma != 0:
        if cfg.args.epsilon != -1:
            filename += f"_private_eps={cfg.args.epsilon}_delta={cfg.args.delta}"
        else:
            filename += f"_private_sigma={cfg.args.sigma}_delta={cfg.args.delta}"
    filename += ".tar"
    return filename

def _load_checkpoint(trainer_cfg, model, device="cpu"): 
    checkpoint_path = _get_checkpoint_path(trainer_cfg)
    print(f"\n====== Attempting to load checkpoint {checkpoint_path} ======")
    checkpoint = {}
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])

        if "epsilon" not in checkpoint:
            checkpoint["epsilon"] = float("inf")
        if "delta" not in checkpoint:
            checkpoint["delta"] = max(0, trainer_cfg.args.delta)
        if "noise_multiplier" not in checkpoint:
            checkpoint["noise_multiplier"] = max(0, trainer_cfg.args.sigma)
        if "steps" not in checkpoint:
            checkpoint["steps"] = -1 # Let CanarySyncTrainer compute this
        if "train_acc" not in checkpoint:
            checkpoint["train_acc"] = 0
        if "test_acc" not in checkpoint:
            checkpoint["test_acc"] = 0
            
        print(f"Checkpointed FL model loaded successfully epoch={checkpoint['epoch']}, round={checkpoint['round']}")
        print(f"Checkpointed model DP guarantees (eps, delta)=({checkpoint['epsilon']}, {checkpoint['delta']}) sigma={checkpoint['noise_multiplier']}")

        # TODO: Rework this?
        trainer_cfg.args.canary_insert_epoch = 1
        trainer_cfg.args.canary_insert_test_acc = -1
        trainer_cfg.args.canary_insert_train_acc = -1
    except FileNotFoundError:
        print("Checkpoint not found for the specific combination of parameters, resorting to training model from scratch")

    return checkpoint

def create_model(model_config, data_config, in_channels, vocab_size, emb_size):
    if model_config.model_arch == "resnet":
        model = Resnet18(num_classes=model_config.num_classes, in_channels=in_channels)
    elif model_config.model_arch == "lstm":
        model = Sent140StackedLSTMModel(
            seq_len=data_config.max_seq_len,
            num_classes=model_config.num_classes,
            emb_size=emb_size,
            n_hidden=model_config.n_hidden,
            vocab_size=vocab_size,
            dropout_rate=model_config.dropout,
        )
    elif model_config.model_arch == "shakes_lstm":
        model = ShakespeareModel(
            seq_len=model_config.seq_len,
            n_hidden=model_config.n_hidden,
            num_classes=model_config.num_classes,
            dropout_rate=model_config.dropout,
        )
    else:
        model = SimpleConvNet(num_classes=model_config.num_classes, in_channels=in_channels, dropout_rate=model_config.dropout)
    
    return model 

def create_data_provider(trainer_config, data_config):
    in_channels, vocab_size, emb_size = 0, 0, 0
    if trainer_config.args.dataset == "CIFAR10":
        data_provider = build_data_provider_cifar10(
            data_root=data_config.data_root,
            local_batch_size=data_config.local_batch_size,
            examples_per_user=data_config.examples_per_user,
            drop_last=False,
            disable_aug=trainer_config.args.prettify_samples
        )
        in_channels = 3
    elif "celeba" in trainer_config.args.dataset:
        data_provider = build_data_provider_celeba(data_config, trainer_config, disable_aug=trainer_config.args.prettify_samples)
        in_channels = 3
    elif "femnist" in trainer_config.args.dataset:
        data_provider = build_data_provider_femnist(data_config, disable_aug=trainer_config.args.prettify_samples)
        in_channels = 1
    elif "shakespeare" in trainer_config.args.dataset:
        data_provider = build_data_provider_shakespeare(data_config)
    else:
        data_provider, vocab_size, emb_size  = build_data_provider_sent140(      
            local_batch_size=data_config.local_batch_size,
            vocab_size=data_config.vocab_size,
            num_users=data_config.num_users,
            user_dist=data_config.user_dist,
            max_seq_len=data_config.max_seq_len,
            drop_last=False,
            data_path=data_config.data_root,
        )    
    return data_provider, in_channels, vocab_size, emb_size


# Main
def main_worker(
    trainer_config,
    data_config,
    model_config,
    use_cuda_if_available: bool = True,
    distributed_world_size: int = 1,
) -> None:
    original_trainer_config = copy.deepcopy(trainer_config) # If loading checkpoints, the trainer config is modified to change canary insert epochs to 1 
    emb_size, vocab_size = 0,0 # For sent140

    checkpoint_path = _get_checkpoint_path(trainer_config)
    if (trainer_config.args.fl_load_checkpoint) and not os.path.isfile(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist, experiment exiting early...")
        return 
    
    if trainer_config.checkpoint_only:
        print(f"Checkpoint only run - will save checkpoint as {checkpoint_path}")

    data_provider, in_channels, vocab_size, emb_size = create_data_provider(trainer_config, data_config)
    
    for exp_num in range(0, data_config.canary_iters):
        torch.cuda.empty_cache()

        trainer_config = copy.deepcopy(original_trainer_config)
        if not data_config.debug_config:
            trainer_config["plot_path"] = get_plot_path(trainer_config.args, exp_num=exp_num, file_suffix="")
            trainer_config["result_path"] = get_plot_path(trainer_config.args, exp_num, ".tar")

        model = create_model(model_config, data_config, in_channels, vocab_size, emb_size)
        print(model)
        
        cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
        device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

        checkpoint = {}
        if trainer_config.load_checkpoint:
           checkpoint = _load_checkpoint(trainer_config, model, device)
                                                          
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
            checkpoint=checkpoint
        )

        if trainer_config.checkpoint_only and not trainer.insert_acc_achieved:
            trainer.logger.info("Failed to achieve insert accuracy, checkpointing model anyway...")
            trainer._checkpoint_model(trainer_config.epochs, 1, final=True)
        
        if not hasattr(trainer, "canary_analyser") and data_config.canary_iters > 1:
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
