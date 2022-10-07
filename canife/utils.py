#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import re
import string
import unicodedata
from typing import List

import torch


# Sent140 character embeddings
class TextProcessorSent140():
    def __init__(self):
        self.all_letters = {c: i for i, c in enumerate(string.printable)}
        self.reverse_map_all_letters = {i: c for i, c in enumerate(string.printable)}
        self.num_letters = len(self.all_letters)
        self.vocab_size = self.num_letters+1
        self.UNK: int = self.num_letters
    
    def unicodeToAscii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in self.all_letters
        )

    def split_line(self, line):
        """split given line/phrase into list of words
        Args:
            line: string representing phrase to be split

        Return:
            list of strings, with each string representing a word
        """
        return re.findall(r"[\w']+|[.,!?;]", line)

    def flatten_list(self, nested_list):
        return list(itertools.chain.from_iterable(nested_list))

    def line_to_indices(self, line: str, max_seq_len: int):
        line_list = self.split_line(line)  # split phrase in words
        line_list = line_list
        chars = self.flatten_list([list(word) for word in line_list])
        # padding
        indices: List[int] = [
            self.all_letters.get(letter, self.UNK)
            for i, letter in enumerate(chars)
            if i < max_seq_len
        ]
        indices = indices + ([self.UNK] * (max_seq_len - len(indices)))
        return indices

    # Assume input is a tensor of indices
    def index_sequence_to_text(self, indices):
        line = ""
        for i in indices:
            line += self.reverse_map_all_letters.get(i.item(), "�")
        return line

    def text_to_index_sequence(self, text):
        return torch.tensor([self.all_letters.get(c, self.UNK) for c in text])

# Preprocessing for Shakespeare
class TextProcessorShakes():
    def __init__(self) -> None:
        self.all_letters = (
            "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        )
        self.vocab_size = len(self.all_letters)
        
    def word_to_indices(self, word):
        """returns a list of character indices
        Args:
            word: string

        Return:
            indices: int list with length len(word)
        """
        indices = []
        for c in word:
            indices.append(self.all_letters.find(c))
        return indices

    def index_sequence_to_text(self, indices):
        line = ""
        for i in indices:
            line += self.all_letters[i]
        return line
    
    def _one_hot(self, index, size):
        """returns one-hot vector with given size and value 1 at given index"""
        vec = [0 for _ in range(size)]
        vec[int(index)] = 1
        return vec

    def letter_to_vec(self, letter):
        """returns one-hot representation of given letter"""
        index = self.all_letters.find(letter)
        return index  # _one_hot(index, NUM_LETTERS)

def get_plot_path(args, exp_num=1, file_suffix=".png"):
    plot_name = args.model_arch + "_" + args.canary_loss + "_B=" + str(args.local_batch_size)
    if args.canary_setup == "holdout":
        plot_name += "_CanaryDesign=" + str(args.canary_design_sample_size) + "_" + str(args.canary_design_minibatch_size)
    plot_name += "_" + args.canary_setup + "_checkpoint_epoch=" + str(args.canary_insert_epoch) + "_iter=" + str(exp_num)
    plot_path = args.dump_path + args.plot_path + "/" + plot_name + file_suffix
    return plot_path

def state_dict_to_cpu(state_dict):
    """Moves a state dict from GPU to CPU

    Args:
        state_dict: model state dict (on GPU)

    Returns:
        state_dict: model state dict (on CPU)
    """
    for k,v in state_dict.items():
        state_dict[k] = v.detach().clone().cpu()
    return state_dict

def display_gpu_mem(device, logger=None, prefix=""):
    """Debug function - displays device GPU memory statistics

    Args:
        device: GPU device
        logger (_type_, optional): Optional logger. Defaults to None.
        prefix (str, optional): Add prefix to debug output. Defaults to "".
    """
    if str(device) != "cpu":
        mem = torch.cuda.mem_get_info(device=device)
        if logger is None:
            print(prefix, torch.cuda.mem_get_info(device=device))
        else:
            logger.debug(f"{prefix} {mem} {round((mem[1] - mem[0]) / 1024**3, 4)}Gb used")

def count_params(model):
    """Counts number of parameters (that require grad) in a model

    Args:
        model: Model to count params

    Returns:
        Total number of parameters (that require grad)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clip_grad(grad, clip_const=1):
    """Clip gradient

    Args:
        grad (tensor): Gradient to clip
        clip_const (int, optional): Clipping constant. Defaults to 1.

    Returns:
        Clipped gradient tensor
    """
    if torch.norm(grad) > clip_const:
        grad = grad*clip_const / torch.norm(grad)
    return grad

def compute_batch_grad(model, criterion, batch, device="cpu"):
    """Computes average gradients of a batch

    Args:
        model: nn.Module
        criterion: Loss function
        batch: Batch to compute average gradients 
        device (str, optional): Torch device. Defaults to "cpu".

    Returns:
        Batch gradients, moved to cpu
    """
    model.to(device)
    model.zero_grad()

    img = batch[0].to(device)
    target = batch[1].to(device)
    outputs = model(img)

    batch_losses = criterion(outputs, target)
    batch_losses.backward()

    batch_grads = torch.tensor([]).to(device)
    for p in model.parameters():
        if p.requires_grad:
            batch_grads = torch.cat([batch_grads, p.grad.detach().clone().flatten()])

    model.zero_grad()
    return batch_grads.cpu()

def compute_sample_grads(grad_sample_module, criterion, batch, device="cpu", move_grads_to_cpu=True, clipping=True, clipping_const=1):
    """Computes per-sample gradients given a GSM and a batch

    Args:
        grad_sample_module: GradSampleModule
        criterion: Loss function
        batch: Batch to compute per-sample grads of
        device (str, optional): Defaults to "cpu".
        move_grads_to_cpu (bool, optional): If True will move all sample grads to cpu. Defaults to True.
        clipping (bool, optional): Whether to clip per-sample-gradients. Defaults to True.
        clipping_const (int, optional): Clipping const. Defaults to 1.

    Returns:
        batch_grads: Per-sample gradients of batch
        clip_count: Number of per-sample gradients that were clipped
    """
    grad_sample_module.to(device)
    grad_sample_module.zero_grad()

    img = batch[0].to(device)
    target = batch[1].to(device)
    outputs = grad_sample_module(img)

    batch_losses = criterion(outputs, target)
    batch_losses.backward()

    batch_grads = torch.hstack([p.grad_sample.detach().clone().view(img.shape[0], -1) for p in grad_sample_module.parameters()])
    
    clip_count = 0
    if clipping:
        for i, grad in enumerate(batch_grads):
            grad_norm = torch.norm(grad)
            if grad_norm > clipping_const:
                clip_count += 1
                batch_grads[i] = batch_grads[i]*clipping_const / grad_norm
    
    # Calling zero-grad of GradSampleModule without DPOptimizer doesn't remove sample grads (?)
    grad_sample_module.zero_grad()
    for p in grad_sample_module.parameters():
        p.grad_sample = None

    if move_grads_to_cpu:
        return batch_grads.cpu(), clip_count
    else:
        return batch_grads, clip_count

def compute_local_update(model, criterion, optimizer, batches, reverse_batch_scaling=True, expected_batch_size=1, compute_sample_grads=False, local_epochs=1, device="cpu"):
    """Computes a model update given a set of local batches 

    Args:
        model: nn.Module
        criterion: Loss function
        optimizer: Model optimizer
        batches: Mock client local batches
        reverse_batch_scaling (bool, optional): Reverse 1/B averaging, multiplies gradients by B/expected B. Defaults to True.
        expected_batch_size (int, optional): The expected batch size. Defaults to 1.
        compute_sample_grads (bool, optional): Whether to also compute per-sample gradients. If True expects model to be a GSM. Defaults to False.
        local_epochs (int, optional): Number of local epochs to perform. Defaults to 1.
        device (str, optional): Defaults to "cpu".

    Returns:
        local_model_state: The model state dict after the local training. Can be used to compute a model update by differencing with global model.
        local_model_before_insert: Local model at step n-1 where n is the number of local batches
        sample_grads: The per-sample grads, defaults to empty tensor is compute_sample_grads=False
    """
    model.to(device)
    sample_grads = torch.tensor([])
    local_model_before_insert = None

    for epochs in range(local_epochs):
        for i, batch in enumerate(batches):
            img, target = batch
            model.zero_grad()
            if i == len(batches)-1:
                local_model_before_insert = state_dict_to_cpu(copy.deepcopy(model.state_dict()))

            img = img.to(device)
            target = target.to(device)
            outputs = model(img)
            batch_losses = criterion(outputs, target)
            batch_losses.backward()

            if reverse_batch_scaling:
                for p in model.parameters():
                    p.grad *= (img.shape[0]/expected_batch_size)

            if compute_sample_grads:
                sample_grads = torch.cat((sample_grads, torch.hstack([p.grad_sample.clone().cpu().view(img.shape[0], -1) for p in model.parameters()])), dim=0)

            optimizer.step()

    model.zero_grad()
    return model.state_dict(), local_model_before_insert, sample_grads