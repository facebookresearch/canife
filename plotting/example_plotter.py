#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import pathlib
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


sys.path.append("../canife")
sys.path.append("../privacy_lint")

from collections import defaultdict

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from canife import CanaryAnalyser
from privacy_lint.privacy_lint.attack_results import AttackResults


BASE_PATH = str(pathlib.Path(__file__).parent.resolve())

sns.set_theme(style="whitegrid")

def set_fontsize(size=14):
    usetex = matplotlib.checkdep_usetex(True)
    tex_fonts = {
        "text.usetex": usetex,
        "font.family": "serif",
        "axes.labelsize": size,
        "font.size": size,
        "legend.fontsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size
    }
    plt.rcParams.update(tex_fonts)
    
FONT_SIZE = 20
set_fontsize(FONT_SIZE)

# convert pandas col names to readable plot labels
column_map = {
    "global_round": r"Global Round ($r$)",
    "empirical_eps_upper": r"$\hat{\varepsilon}_U$",
    "empirical_eps_lower": r"$\hat{\varepsilon}_L$",
    "empirical_eps": r"$\hat{\varepsilon}$",
    "current_test_acc": r"Model Test Accuracy",
    "current_train_acc": r"Model Train Accuracy",
    "canary_health": r"Canary Health",
    "mia_acc": r"Attack Accuracy ($\gamma = 0.5$)",
    "mia_max_acc": r"Attack Accuracy",
    "mia_max_acc_rolling": r"Attack Accuracy",
    "acc_rolling": r"Attack Accuracy",
    "final_epsilon": r"Privacy Budget ($\varepsilon$)",
    "one_step_eps": r"One-step $\varepsilon$",
    "num_clients": r"Clients Per Round",
    "canary_design_pool_size": r"Design Pool Size ($m$)",
    "canary_design_sample_size": "Design Sample Size",
    "average_sd": r"Mean Standard Deviation",
    "with_canary_sd": r"With Canary SD",
    "without_canary_sd": r"Without Canary SD",
    "mia_auc": r"Attack AUC",
    "empirical_global_eps": r"$\hat{\varepsilon}}$",
    "epsilon": r"$\varepsilon$",
    "canary_epochs": r"Design Iterations ($t$)",
    "canary_norm_constant": r"Canary Gradient Norm Constant",
    "dataset": "Dataset"
}

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', len(x.columns))
    print(x)
    pd.reset_option('display.max_rows')
    
def format_axis(ax):
    xlabel = ax.xaxis.get_label()._text
    ylabel = ax.yaxis.get_label()._text
    
    xlabel = column_map.get(xlabel, xlabel)
    ylabel = column_map.get(ylabel, ylabel)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def save_plot(name="", fig=None):
    plt.tight_layout()
    if fig:
        fig.savefig(f"{BASE_PATH}/{name}.pdf", bbox_inches='tight', format="pdf")
    else:
        plt.savefig(f"{BASE_PATH}/{name}.pdf", bbox_inches='tight', format="pdf")
    plt.clf()

def extract_epsilon_metrics(df, override_empirical_eps=False, use_max_acc=False):
    extra_cols = defaultdict(list)
    if override_empirical_eps:
        print("Extracting empirical epsilon data...")
        analyser = CanaryAnalyser(None, None, None)
        
        for idx, x in df.iterrows():
            with_dot_prods = ast.literal_eval(x["dot_prods_with_canary"].replace('nan,', ''))
            without_dot_prods = ast.literal_eval(x["dot_prods_without_canary"].replace('nan,', ''))
            results = AttackResults(torch.tensor(with_dot_prods), torch.tensor(without_dot_prods))
            
            max_acc_thresh = results.get_max_accuracy_threshold()[0]
            n_pos, n_neg = len(results.scores_train), len(results.scores_test)            
            max_empirical_eps = 0
            
            _, scores = results._get_scores_and_labels_ordered()
            tpr_fpr = results.get_tpr_fpr()
            
            # delta = 1e-5
            delta = 1/(n_pos + n_neg)
            
            if use_max_acc: # Calculate empirical eps from max acc threshold
                tp = int((results.scores_train >= max_acc_thresh).sum().item())
                fp = int((results.scores_test >= max_acc_thresh).sum().item())
                max_fp, max_fn, max_tp, max_tn = fp, n_pos-tp, tp, n_neg-fp
                max_tpr, max_fpr = max_tp / (max_tp + max_fn), max_fp/(max_fp+max_tn)
                max_empirical_eps = analyser.empirical_eps(max_tpr, max_fpr, delta=delta)
            else: # Maximise empirical eps over TPR/FPR
                for i, t in enumerate(scores):
                    tpr, fpr = tpr_fpr[0][i], tpr_fpr[1][i]
                    empirical_eps = analyser.empirical_eps(tpr, fpr, delta=delta)
                    acc = results.get_accuracy(t)
                    
                    if empirical_eps > max_empirical_eps and (empirical_eps != float("inf") or acc == 1):
                        tp = int((results.scores_train >= t).sum().item())
                        fp = int((results.scores_test >= t).sum().item())
                        max_empirical_eps = empirical_eps
                        max_fp, max_fn, max_tp, max_tn = fp, n_pos-tp, tp, n_neg-fp
                        max_tpr, max_fpr = tpr, fpr
            
            lower_eps = analyser.ci_eps(max_fp, max_fn, n_pos, n_neg, delta=delta)
            upper_eps = analyser.ci_eps(max_fp, max_fn, bound="upper", n_pos=n_pos, n_neg=n_neg, delta=delta)
                        
            extra_cols["fp"].append(max_fp)
            extra_cols["fn"].append(max_fn)
            extra_cols["tp"].append(max_tp)
            extra_cols["tn"].append(max_tn)
            extra_cols["empirical_eps_lower"].append(lower_eps)
            extra_cols["empirical_eps_upper"].append(upper_eps)
            extra_cols["empirical_eps"].append(max_empirical_eps)

        for col in extra_cols.keys():
            df[col] = extra_cols[col]
            
        print("Empirical epsilon data added...")

def extract_global_empirical_eps(df, skip_ci=True):
    df["empirical_global_eps"] = 0
    df["empirical_global_eps_lower"] = 0
    df["empirical_global_eps_upper"] = 0
    df = df.sort_values(by="global_round")
    
    eps_list = df["epsilon"].unique()
    sample_rate = df["sample_rate"].unique()[0]
    print(f"Eps list {eps_list}, sample rate={sample_rate}")
    for eps in eps_list:
        temp_df = df[df["epsilon"] == eps]
        temp_df = temp_df.sort_values(by="global_round")
        df_eps = temp_df["empirical_eps"].values
        steps = temp_df["global_round"].values
        
        theoretical_sigma = temp_df["final_sigma"].mean()

        empirical_global_eps = calculate_global_eps(df_eps, theoretical_sigma=theoretical_sigma, steps=steps, sample_rate=sample_rate)
        df.loc[df["epsilon"] == eps, 'empirical_global_eps'] = empirical_global_eps
        print(f"eps={eps} estimate done...")

        if not skip_ci:
            empirical_global_eps = calculate_global_eps(temp_df["empirical_eps_lower"].clip(lower=0.2).values, theoretical_sigma=theoretical_sigma, steps=steps, sample_rate=sample_rate)
            df.loc[df["epsilon"] == eps, 'empirical_global_eps_lower'] = empirical_global_eps
            print(f"eps={eps} lower done...")

            empirical_global_eps = calculate_global_eps(temp_df["empirical_eps_upper"].values, theoretical_sigma=theoretical_sigma, steps=steps, sample_rate=sample_rate)
            df.loc[df["epsilon"] == eps, 'empirical_global_eps_upper'] = empirical_global_eps
            print(f"eps={eps} upper done...\n")
            
    return df

def compute_one_step_eps(sample_rate, noise, delta=1e-5):
    accountant = RDPAccountant()
    history_step = (noise, sample_rate, 1)
    accountant.history.append(history_step)
    current_budget = accountant.get_privacy_spent(delta=delta)
    return current_budget[0]
            
def calculate_global_eps(empirical_per_step_epsilons, theoretical_sigma, sample_rate=0.01, steps=1000, delta=1e-5, n=100, verbose=False):
    if type(steps) == int:
        steps = range(1, steps+1)
            
    accountant = RDPAccountant()
    previous_step = 0
    if verbose:
        theoretical_accountant = RDPAccountant()
        one_step_theoretical_eps = compute_one_step_eps(1, theoretical_sigma, delta)

    budgets = []
    for i,step in enumerate(steps):
        # One-step sigma based on current empirical eps
        if empirical_per_step_epsilons[i] == float("inf"): # Resort to theoretical sigma if empirical eps is inf
            estimated_sigma = theoretical_sigma
        else:
            estimated_sigma = get_noise_multiplier(target_epsilon=max(empirical_per_step_epsilons[i], 0.15),
                                                target_delta=1/n,
                                                sample_rate=1,
                                                steps=1)
        
        # Assume noise is constant for step-previous_step rounds (i.e time between last estimate and current)
        history_step = (estimated_sigma, sample_rate, step-previous_step)
        accountant.history.append(history_step)
        previous_step = step
        current_budget = accountant.get_privacy_spent(delta=delta)
        budgets.append(current_budget[0])
        
        if verbose:
            estimated_sigma_theoretical = get_noise_multiplier(target_epsilon=one_step_theoretical_eps,
                                                target_delta=delta,
                                                sample_rate=1,
                                                steps=1)
            history_step = (estimated_sigma_theoretical, sample_rate, step-previous_step)
            theoretical_accountant.history.append(history_step)
            theoretical_eps = theoretical_accountant.get_privacy_spent(delta=delta)[0]
        
            print(f"i={i}, global round={step}")
            print(f"Estimated empirical one-step sigma = {estimated_sigma} vs theoretical = {estimated_sigma_theoretical}")
            print(f"Estimated empirical one-step epsilon = {empirical_per_step_epsilons[i]} vs theoretical = {one_step_theoretical_eps}")
            print(f"Accumulated empirical budget {budgets[-1]} vs theoretical {theoretical_eps}\n")
    
    return budgets

def load_sweep(name, relative_path=False, override_empirical_eps=False):
    if relative_path:
        df = pd.read_csv(name)
    else:
        df = pd.read_csv(BASE_PATH + "/" + name + ".csv")
        
    df["sd_gap"] = np.sqrt(df["without_canary_var"]) - np.sqrt(df["with_canary_var"])
    print(df.columns)
    
    # For compatability with old sweeps where some metrics were tensors
    if df["mia_acc"].dtype == object:
        for s in ["tensor", "(", ")"]:
            df["mia_acc"] = df["mia_acc"].str.replace(s, "")
        df["mia_acc"] = df["mia_acc"].astype("float64")
    
    extract_epsilon_metrics(df, override_empirical_eps=override_empirical_eps)

    return df

def plot(csv_name):
    dataset = "sent140"
    model = "lstm"
    xlim = 8900
    main_df = load_sweep(f"{csv_name}", relative_path=True, override_empirical_eps=False)
    main_df["epsilon"] = main_df["epsilon"].astype("int")
    main_df = main_df[main_df["dataset"] == dataset]
    
    main_df = main_df[main_df["epsilon"].isin([10,30,50])]
    
    # Per-round empirical eps comparison
    for eps in main_df["epsilon"].unique():
        plot_df = main_df.copy()
        plot_df = plot_df[plot_df["epsilon"] == eps]
        plot_df.replace([float("inf")], np.nan, inplace=True)
        for y in ["one_step_eps"]:
            plot_df = main_df.copy()
            plot_df = plot_df[plot_df["epsilon"] == eps]
            ax = sns.lineplot(data=plot_df, x="global_round", y="empirical_eps", markers=False, label=r"$\hat{\varepsilon}_r$")
            plt.fill_between(plot_df["global_round"].values, plot_df["empirical_eps_lower"].values, plot_df["empirical_eps_upper"].values, alpha=.3)
            sns.lineplot(data=plot_df, x="global_round", y=y, markers=False, label=r"$\varepsilon_r$", ax=ax)
            plt.ylim(0)
            plt.xlim(0, xlim)
            plt.tight_layout()
            plt.draw()
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95)) 
            format_axis(ax)
            ax.set_ylabel(r"Privacy Budget ($\varepsilon$)")
            save_plot(name=f"{dataset}_{eps}_{model}_per_round_eps")
    
    # Global empirical eps comparison
    plot_df = main_df.copy()
    
    main_palette = sns.color_palette("deep", 3)
    palette_dict = {10: main_palette[0], 30: main_palette[1], 50: main_palette[2]}
    palette = [palette_dict[eps] for eps in plot_df["epsilon"].unique()]
    ax = sns.lineplot(data=plot_df, x="global_round", y="final_epsilon", hue="epsilon", linestyle="--", palette=palette)
    sns.lineplot(data=plot_df, x="global_round", y="empirical_global_eps", hue="epsilon", ax=ax, label='_nolegend_', palette=palette)
    plt.xlim(0, xlim)
    format_axis(ax)
    hand, labl = ax.get_legend_handles_labels()
    handout=[]
    lablout=[]
    for h,l in zip(hand,labl):
       if l not in lablout:
            lablout.append(l)
            handout.append(h)
    legend1 = plt.legend(handout, lablout, title=r"$\varepsilon$")
    plt.ylim(0, 50)

    linestyles = ['-', "--"]
    dummy_lines = []
    titles = [r"Empirical $\hat{\varepsilon}$", r"Theoretical $\varepsilon$"]
    for b_idx, b in enumerate(titles):
        dummy_lines.append(ax.plot([],[], c="black", ls = linestyles[b_idx])[0])
    plt.legend([dummy_lines[i] for i in [0,1]], titles, loc="upper left", bbox_to_anchor=(0.4,0.6))
    ax.add_artist(legend1)
    save_plot(name=f"{dataset}_global_eps")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot example canife experiment")
    parser.add_argument("--csv-path", type=str, help= "Path to output .csv for plotting")

    args = parser.parse_args()
    global_eps_path = args.csv_path.split(".csv")[0] + "_extracted.csv"
    
    global_eps_csv_file = Path(global_eps_path)
    csv_file = Path(args.csv_path)
        
    if not csv_file.is_file():
        raise FileNotFoundError(f"Output .csv does not exist at the given file path {args.csv_path}")
        
    if not global_eps_csv_file.is_file():
        df = pd.read_csv(args.csv_path)
        df = extract_global_empirical_eps(df)
        df.to_csv(global_eps_csv_file)
        
    plot(csv_name=global_eps_csv_file)