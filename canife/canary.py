#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

class Canary():
    def __init__(self, data, init_data, class_label, init_loss=0, init_grad=None, canary_grad=None, final_loss=0, health=0) -> None:
        """Canary class

        Args:
            data: Tensor of final optimised canary
            init_data: Tensor of initial canary (before optimisation)
            class_label: Canary class
            init_loss (int, optional): Initial canary loss. Defaults to 0.
            init_grad (tensor, optional): Initial canary gradient. Defaults to None.
            canary_grad (tensor, optional): Final canary gradient. Defaults to None.
            final_loss (int, optional): Final loss after optimisation. Defaults to 0.
            health (int, optional): Canary health between 0-1. Defaults to 0.
        """
        self.data = data
        self.init_data = init_data

        self.final_loss = final_loss
        self.init_loss = init_loss 

        self.class_label = class_label
        self.health = health

        self.grad = canary_grad
        self.init_grad = init_grad
        self.health = health