#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os

import pandas as pd
import torch


USERNAME = os.getlogin()
print(f"USERNAME: {USERNAME}")

def extract_sweep(root_dir="saved_sweeps", csv_name=""):
    rows = []
    full_path = root_dir 
    tar_path = full_path + "/**/*.tar"

    print("Full path", full_path)

    for file in glob.glob(tar_path, recursive=True):
        exp_checkpoint = torch.load(file)
        row = exp_checkpoint["row"]
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

    save_path = f"/checkpoints/{USERNAME}/" + csv_name + ".csv"
    df.to_csv(f"/checkpoints/{USERNAME}/" + csv_name + ".csv")
    print(f"Sweep extracted saved to {save_path}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract canife sweep")
    parser.add_argument("--sweep", type=str, help= "Name of saved sweep")
    args = parser.parse_args()
    extract_sweep(csv_name=args.sweep, root_dir=f"/checkpoints/{USERNAME}/canife/{args.sweep}")