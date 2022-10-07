#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import matplotlib.pyplot as plt


sys.path.append("./FLSim")
from arg_handler import create_flsim_cfg, parse_args
from FLSim.examples.canary_example import run
from FLSim.flsim.common.logger import Logger


plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("ddp")
logger.setLevel(level=logging.INFO)

num_class_map = {"CIFAR10": 10, "imagenet": 1000, "sent140": 2, "femnist": 62, "celeba": 2, "shakespeare": 80}

# ----------------- Args + Main -----------------

if __name__ == "__main__":
    args = parse_args()

    if not args.debug_config:
        args.canary_design_minibatch_size  = int(args.users_per_round) if args.canary_design_minibatch_size == "num_users" else args.canary_design_minibatch_size
        args.canary_design_pool_size  = int(args.users_per_round) if args.canary_design_pool_size == "num_users" else args.canary_design_pool_size
        if args.canary_design_type == "sample_grads": # Defaults for sample grads
            if args.canary_design_pool_size != "": # Design pool size overrides design sample size
                args.canary_design_sample_size = args.canary_design_pool_size
            else:
                args.canary_design_sample_size = 32 if args.canary_design_minibatch_size == "" else args.canary_design_minibatch_size
                args.canary_design_pool_size = args.canary_design_sample_size
            args.canary_design_minibatch_size  = args.canary_design_sample_size  if args.canary_design_minibatch_size == "" else args.canary_design_minibatch_size
            args.local_batch_size = 128 if args.local_batch_size == "" else args.local_batch_size
        else: # Defaults for model_updates
            args.local_batch_size = 128 if args.local_batch_size == "" else args.local_batch_size
            if args.canary_design_minibatch_size == "":
                args.canary_design_minibatch_size  = int(args.users_per_round) if args.canary_design_type == "model_updates" else int(args.local_batch_size)
            args.canary_design_sample_size = int(args.local_batch_size) * abs(args.num_local_updates) * int(args.canary_design_minibatch_size) if args.canary_design_sample_size == "" else args.canary_design_sample_size
            if args.canary_design_pool_size != "":
                args.canary_design_sample_size = int(args.canary_design_pool_size) * abs(args.num_local_updates) * int(args.local_batch_size)
        
        args.canary_design_sample_size = int(args.canary_design_sample_size)
        args.canary_design_minibatch_size = int(args.canary_design_minibatch_size)
        args.local_batch_size = int(args.local_batch_size)
        args.canary_design_pool_size = int(args.canary_design_pool_size) if args.canary_design_pool_size != "" else -1
            
    args.num_classes = num_class_map[args.dataset]
    if args.task == "FLSim": # Run FLSim with a canary attack
        # Load config and run flsim
        if args.debug == 1:
            Logger.set_logging_level(logging.DEBUG)
        cfg = create_flsim_cfg(args)
        print(args.dataset)
        run(cfg)