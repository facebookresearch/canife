#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob

import pandas as pd
import torch


def extract_sweep(root_dir="local_checkpoints", csv_name=""):
    rows = []
    full_path = root_dir 
    tar_path = full_path + "/**/*.tar"

    print("Full path", full_path)

    for file in glob.glob(tar_path, recursive=True):
        exp_checkpoint = torch.load(file)
        row = exp_checkpoint["row"]
        # row.append(exp_checkpoint["batch_clip_percs"])
        columns = exp_checkpoint["columns"]
        columns.extend(["train_acc", "test_acc"])
        row.extend([-1,-1])
        
        if "accuracy_metrics" in columns:
            metrics = row[columns.index("accuracy_metrics")]

            if len(metrics["train"]) > 0:
                train_acc = metrics["train"][-1]
                row[-2] = train_acc

            if len(metrics["test"]) > 0:
                test_acc = metrics["test"][-1]
                row[-1] = test_acc
            
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    print(df.info(memory_usage="deep"))

    save_path = f"{args.path}{args.csv_name}"
    df.to_csv(save_path)
    print(f"Sweep extracted saved to {save_path}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract canife experiment")
    parser.add_argument("--path", type=str, help= "Path to location of experiment output")
    parser.add_argument("--csv-name", type=str, help= "Name of output .csv")

    args = parser.parse_args()
    extract_sweep(csv_name=args.csv_name, root_dir=args.path)