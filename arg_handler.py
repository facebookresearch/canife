#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import sys


sys.path.append("./FLSim")
from FLSim.flsim.utils.config_utils import fl_config_from_json


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ["False", "false"]:
        return False
    elif s.lower() in ["True", "true"]:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def flsim_args(parser):
    parser.add_argument(
        "--dp-level",
        default="user_level",
        type=str,
        help="FLSim DP level (User or item level DP). Defaults to user_level.",
    )

    parser.add_argument(
        "--gpu-mem-minimiser",
        default="False",
        type=bool_flag,
        help="FLSim, whether to use the GPUMemoryMinimiser",
    )

    parser.add_argument(
        "--debug-config",
        default="False",
        type=bool_flag,
        help="For debugging: Whether to use FLSim debug configs (without CanarySyncTrainer)",
    )

    parser.add_argument(
        "--users-per-round",
        default=1,
        type=int,
        help="FLSim, Sets the number of users per round for training + attacking FL models",
    )

    parser.add_argument(
        "--client-epochs",
        default=1,
        type=int,
        help="FLSim, number of local epochs per user",
    )

    parser.add_argument(
        "--num-local-updates",
        default=-1,
        type=int,
        help="FLSim, number of local updates made by a user. -1 if users have varying number of local batches (default)",
    )

    parser.add_argument(
        "--server-clip-const",
        default=1,
        type=int,
        help="Sets the FLSim 'clipping_value' parameter. This is the clipping constant of model updates.",
    )

    parser.add_argument(
        "--canary-design-reverse-server-clip",
        default=False,
        type=bool_flag,
        help="For debugging: If True, will design and test on unclipped server updates, but will still train the model on clipped server updates",
    )

    parser.add_argument(
        "--insert-canary-as-batch",
        default=False,
        type=bool_flag,
        help="Whether to insert the canary as a sample or an entire batch. Does not need to be set, will be updated based on canary-insert-batch-index",
    )

    parser.add_argument(
        "--canary-insert-global-round",
        default=-1,
        type=int,
        help="FLSim, the global round to insert the canary into, overrides canary-insert-epoch",
    )

    parser.add_argument(
        "--canary-insert-offset",
        default=1,
        type=int,
        help="FLSim, used in train_and_freeze and continuous testing and is the round period between attacks",
    )

    parser.add_argument(
        "--canary-insert-batch-index",
        default="batch",
        type=str,
        help="FLSim, the batch index to insert the canary. Options: 0,-1, 'batch', Default: batch (i.e inserts canary on its own)",
    )

    parser.add_argument(
        "--canary-design-local-models",
        type=bool_flag,
        default=False,
        help="For debugging: If True and canary_insert_batch_index=-1, then design canaries on the (num_local_updates-1)th model",
    )
    
    parser.add_argument(
        "--canary-insert-train-acc",
        default=-1,
        type=int,
        help="In FLSim, inserts canary after model achieves train acc >= canary-insert-train-acc, overrides canary-insert-epoch and canary-insert-global-round",
    )

    parser.add_argument(
        "--canary-insert-test-acc",
        default=-1,
        type=int,
        help="In FLSim, inserts canary after model achieves given test acc, overrides canary-insert-epoch, canary-insert-global-round and canary-insert-train-acc",
    )

    parser.add_argument(
        "--canary-insert-type",
        default="",
        type=str,
        help="Types: train (acc), test (acc)",
    )

    parser.add_argument(
        "--canary-test-type",
        default="freeze",
        type=str,
        help="Takes values: 'freeze', 'train_and_freeze', 'continuous'",
    )

    parser.add_argument(
        "--canary-insert-acc-threshold",
        default=-1,
        type=int,
        help="FLSim, Round or accuracy to design canary at and begin CANIFE attack",
    )

    parser.add_argument(
        "--canary-insert-epsilon",
        default=-1,
        type=float,
        help="FLSim, train model to target epsilon before inserting canary, Default: -1 (disabled)",
    )

    parser.add_argument(
        "--epsilon",
        default=-1,
        type=float,
        help="FLSim, will calibrate noise_multiplier to guarantee epsilon over fl-epochs Default -1 (disabled)",
    )

    parser.add_argument(
        "--fl-server-lr",
        default=-1,
        type=float,
        help="FLSim server lr, Default: -1 (uses FLSim config default)",
    )

    parser.add_argument(
        "--fl-client-lr",
        default=-1,
        type=float,
        help="FLSim client lr, Default: -1 (uses FLSim config default)",
    )

    parser.add_argument(
        "--fl-dropout",
        default=0,
        type=float,
        help="FLSim, model dropout if using simpleconv, Default: 0 (no dropout)",
    )

    parser.add_argument(
        "--fl-checkpoint-only",
        default=False,
        type=bool_flag,
        help="FLSim, Train until canary insertion, save checkpoint and then exit",
    )

    parser.add_argument(
        "--fl-load-checkpoint",
        default=False,
        type=bool_flag,
        help="FLSim, Attempt to load the checkpoint of the experiments parameters if possible, otherwise train from scratch",
    )

    parser.add_argument(
        "--fl-epochs",
        default=-1,
        type=int,
        help="FLSim number of epochs Default: -1 (uses FLSim config epochs)",
    )

    parser.add_argument(
        "--local-batch-size",
        default="",
        type=str,
        help="FLSim, Local batch size of FLSim clients",
    )
    
    parser.add_argument(
        "--override-noise-multiplier",
        default="False",
        type=bool_flag,
        help="FLSim, If True, will override noise multiplier with epsilon/sigma even when loading a DP checkpoint",
    )

def canary_args(parser):
    parser.add_argument(
        "--canary-normalize-optim-grad",
        default="True",
        type=bool_flag,
        help="Normalize grad",
    )
    
    # Takes values: Random, Image, Text
    parser.add_argument(
        "--canary-init",
        default="random",
        type=str,
        help="CANIFE, Method for initialising the canary sample. Default: Randomly initialised (from token space or image space)",
    )

    parser.add_argument(
        "--canary-epochs",
        default=5000,
        type=int,
        help="CANIFE, number of canary design iterations",
    )

    parser.add_argument(
        "--canary-iters",
        default=1,
        type=int,
        help="How many times to repeat the canary experiment. Default: 1",
    )

    parser.add_argument(
        "--canary-clip-const",
        default=1,
        type=float,
        help="CANIFE, Canary sample-grad clip factor. Only used for debugging.",
    )

    # loss1 - Square dot product with batch mean
    # loss2 - Square dot product with per sample gradients
    parser.add_argument(
        "--canary-loss",
        default="loss2",
        type=str,
        help="CANIFE, Canary loss to use. Defaults to loss2 (First term of Eq1 in paper)",
    )

    parser.add_argument(
        "--canary-norm-matching",
        default="True",
        type=bool_flag,
        help="CANIFE, If True, will optimise canary sample to have gradient matched to canary-norm-constant",
    )
    
    parser.add_argument(
        "--canary-norm-loss",
        default="hinge_squared",
        type=str,
        help="For debugging: hinge vs hinge_squared",
    )
    
    parser.add_argument(
        "--canary-norm-constant",
        default=1,
        type=int,
        help="CANIFE, If canary_norm_matching=True, will optimise canary to have norm >= canary-norm-consant",
    )

    # sample_grads = Orthogonal to sample grads
    # model_updates = Orthogonal to model updates
    parser.add_argument(
        "--canary-design-type",
        default="model_updates",
        type=str,
        help="CANIFE, whether to design on clipped model updates or on clipped sample grads. Default: model_updates",
    )

    # freeze / holdout
    # exact
    parser.add_argument(
        "--canary-setup",
        default="exact",
        type=str,
        help="CANIFE, Whether to form the design pool of mock clients from a holdout (test) set or 'exact' (design on current rounds clients)",
    )

    parser.add_argument(
        "--canary-insert-epoch",
        default="1",
        type=str,
        help="FLSim, Epoch to design canary from and carry out CANIFE attack",
    )

    parser.add_argument(
        "--canary-num-test-batches",
        default=50,
        type=int,
        help="Number of batches (from the training set) to test canary against",
    )

    parser.add_argument(
        "--canary-design-sample-size",
        default="",
        type=str,
        help="CANIFE, Design pool sample size. If empty will be inferred from canary-design-minibatch-size",
    )

    parser.add_argument(
        "--canary-design-pool-size",
        default="",
        type=str,
        help="CANIFE, Design pools size. If not empty and using model updates, will override sample size",
    )

    parser.add_argument(
        "--canary-design-minibatch-size",
        default="",
        type=str,
        help="CANIFE, Design optimisation minibatch size. If empty will be set to canary_design_sample_size or users_per_round",
    )
    
    parser.add_argument(
        "--benchmark-design",
        default="False",
        type=bool_flag,
        help="CANIFE, Whether to track canary design time or not. Default: False",
    )
    
    parser.add_argument(
        "--scale-canary-test",
        default="False",
        type=bool_flag,
        help="CANIFE, Debugging"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Mad Canaries")
    canary_args(parser)
    flsim_args(parser)

    parser.add_argument(
        "--task",
        default="FLSim",
        type=str,
        help="Task",
    )

    parser.add_argument(
        "--model-arch",
        default="simpleconv",
        type=str,
        help="Model arch options: lstm, resnet, simpleconv, shakes_lstm",
    )

    parser.add_argument(
        "--num-classes",
        default=10,
        type=int,
        help="",
    )
    
    parser.add_argument(
        "--sigma",
        type=float,
        default=0,
        metavar="S",
        help="Noise multiplier for DP (default 0)",
    )
    
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target DP delta (default: 1e-5)",
    )

    parser.add_argument(
        "--disable-dp",
        type=bool_flag,
        default=False,
        help="Not used in FLSim/CANIFE. Disable privacy training and just train with vanilla SGD.",
    )

    parser.add_argument(
        "--skip-acc",
        type=bool_flag,
        default=False,
        help="If True, does not benchmark accuracy when loading a checkpointed model in central canary attack",
    )

    parser.add_argument(
        "--checkpoint",
        type=bool_flag,
        default=True,
        help="Save checkpoints every checkpoint_round during training",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./local_checkpoints",
        help="path of checkpoints (saving/loading)",
    )

    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help="Will output experiment results to DUMP_PATH/PLOT_PATH. Default: '' ",
    )

    parser.add_argument(
        "--dump-path",
        type=str,
        default="./local_checkpoints",
        help="Output path of experiment run.",
    )

    parser.add_argument(
        "--checkpoint-round",
        type=int,
        default=5,
        metavar="k",
        help="Not used. FLSim, Checkpoint every k rounds",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Options: CIFAR10, celeba, shakespeare, sent140",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Location of LEAF datsets or CIFAR10",
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device on which to run the code. Values: cpu or gpu"
    )

    parser.add_argument(
        "--master-port",
        default=12568,
        type=str,
        help="Slurm master port",
    )
    
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )
    
    parser.add_argument(
        "--prettify-samples",
        type=bool_flag,
        default="False",
        help="CANIFE, For debugging. Disables data augmentation + outputs canary samples",
    )

    return parser.parse_args()

def create_flsim_cfg(args, base_config="./FLSim/examples/configs/"):    
    config_map = {
        "CIFAR10_True": "cifar10_resnet_canary_sample_level.json",
        "CIFAR10_False": "cifar10_resnet_canary_user_level.json",
        "celeba_True": "celeba_example.json",
        "celeba_False": "celeba_resnet_canary_user_level.json",
        "sent140_True": "sent140_config.json",
        "sent140_False": "sent140_canary_user_level.json",
        "femnist_False": "femnist_config.json",
        "shakespeare_False": "shakespeare_config.json"
    }
    config_key = f"{args.dataset}_{args.debug_config}"
    config_name = config_map.get(config_key, None)
    if config_name is None:
        raise Exception("No viable config provided")
    base_config += config_name

    with open(base_config, "r") as config_file:
        json_config = json.load(config_file)
        if args.dp_level == "server_level":
            json_config["config"]["trainer"]["server"]["privacy_setting"]["clipping_value"] = args.flsim_server_clip_const 
        cfg = fl_config_from_json(json_config["config"])

    if args.canary_insert_type != "":
        if args.canary_insert_type == "train":
            args.canary_insert_train_acc = args.canary_insert_acc_threshold
        elif args.canary_insert_type == "test":
            args.canary_insert_test_acc = args.canary_insert_acc_threshold

    if args.canary_insert_batch_index == "batch":
        args.insert_canary_as_batch = True
    else:
        args.canary_insert_batch_index = int(args.canary_insert_batch_index)

    # Data args
    if args.local_batch_size != "":
        cfg["data"]["local_batch_size"] = int(args.local_batch_size)
    if args.dataset == "CIFAR10":
        cfg["data"]["examples_per_user"] = max(args.local_batch_size, 1)*max(args.num_local_updates,1)
    cfg["data"]["data_root"] = args.data_root
    cfg["data"]["canary_iters"] = args.canary_iters
    cfg["data"]["debug_config"] = args.debug_config

    # Model args
    cfg["model"]["model_arch"] = args.model_arch
    cfg["model"]["dropout"] = args.fl_dropout

    # Trainer args
    cfg["trainer"]["checkpoint_only"] = args.fl_checkpoint_only
    cfg["trainer"]["load_checkpoint"] = args.fl_load_checkpoint
    if not args.debug_config:
        args.canary_insert_epoch = int(args.canary_insert_epoch)
        dict_args = copy.deepcopy(vars(args))

        cfg["trainer"]["users_per_round"] = args.users_per_round
        cfg["trainer"]["args"] = dict_args

    cfg["trainer"]["client"]["epochs"] = args.client_epochs

    if args.fl_server_lr != -1:
        cfg["trainer"]["server"]["server_optimizer"]["lr"] = args.fl_server_lr
    if args.fl_client_lr != -1:
        cfg["trainer"]["client"]["optimizer"]["lr"] = args.fl_client_lr

    if "privacy_setting" in cfg["trainer"]["server"]:
        cfg["trainer"]["server"]["privacy_setting"]["clipping_value"] = args.server_clip_const
        cfg["trainer"]["server"]["privacy_setting"]["target_delta"] = args.delta
        cfg["trainer"]["server"]["privacy_setting"]["noise_multiplier"] = args.sigma

    if args.fl_epochs != -1:
        cfg["trainer"]["epochs"] = args.fl_epochs
        
    if args.canary_test_type == "train_and_freeze" and args.epsilon > 0:
        cfg["trainer"]["always_keep_trained_model"] = True

    return cfg
