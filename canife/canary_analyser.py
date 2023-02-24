#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from opacus import GradSampleModule
from scipy.stats import binomtest

from canife.utils import TextProcessorSent140, TextProcessorShakes, compute_sample_grads
from privacy_lint.privacy_lint.attack_results import AttackResults


class CanaryAnalyser():
    def __init__(self, plot_path, result_path, grad_sample_module=None, canary_epochs=1000, canary_loss="loss1", canary_norm_matching=None, canary_design_type="sample_grads", canary_setup="holdout", canary_init="random",
                    canary_design_minibatch_size=0, canary_design_sample_size = 0, canary_design_pool_size=0, local_batch_size=128, canary_clip_const=1, canary_insert_train_acc=0, canary_insert_test_acc=0, canary_losses=None, canary_norms=None,
                    canary_design_reverse_server_clip=False, canary_design_bias=0, canary_insert_epoch="unknown", canary_insert_global_round=-1, canary_insert_batch_index=-1, canary_insert_acc_threshold=-1, canary_normalize_optim_grad=True,
                    canary_design_local_models=False, local_updates=1, local_epochs=1, canary_type="image", delta=1e-5, sigma=0, epsilon=float('inf'), sample_rate=1, checkpoint_train_acc = 0, checkpoint_test_acc = 0,
                    model_arch="unknown", dataset="unknown", task="canary_attack", dp_level="sample_level", logger=None, benchmark_times=None, server_clip_const=1,
                    actual_sample_size=0, actual_pool_size=0 , actual_minibatch_size=0, canary_norm_constant=1, canary_norm_loss="hinge_squared", scale_canary_test=False, **kwargs) -> None:

        self.reset()

        self.epsilon = epsilon
        self.delta = delta
        self.sigma = sigma
        self.sample_rate = sample_rate
        
        self.global_round = 0

        self.canary_type = canary_type
        self.canary_loss = canary_loss
        self.canary_losses = canary_losses
        self.canary_norms = canary_norms
        self.canary_epochs = canary_epochs
        self.canary_init = canary_init
        self.canary_design_type = canary_design_type
        self.canary_setup = canary_setup
        self.canary_clip_const = canary_clip_const

        self.canary_design_minibatch_size = canary_design_minibatch_size
        self.canary_design_sample_size = canary_design_sample_size
        self.canary_design_pool_size = canary_design_pool_size
        self.scale_canary_test = scale_canary_test
        
        self.actual_sample_size = actual_sample_size
        self.actual_pool_size = actual_pool_size
        self.actual_minibatch_size =  actual_minibatch_size
        
        self.canary_design_reverse_server_clip = canary_design_reverse_server_clip
        self.canary_design_bias = canary_design_bias
        self.local_batch_size = local_batch_size
        self.canary_norm_matching = canary_norm_matching
        self.canary_norm_constant = canary_norm_constant
        self.canary_norm_loss = canary_norm_loss

        self.canary_normalize_optim_grad = canary_normalize_optim_grad
        
        self.model_arch = model_arch
        self.dataset = dataset

        self.canary_insert_epoch = canary_insert_epoch
        self.canary_insert_global_round = canary_insert_global_round
        self.canary_insert_batch_index = canary_insert_batch_index
        self.canary_insert_train_acc = canary_insert_train_acc
        self.canary_insert_test_acc = canary_insert_test_acc
        self.canary_insert_acc_threshold = canary_insert_acc_threshold

        self.canary_design_local_models = canary_design_local_models
        self.local_updates = local_updates
        self.local_epochs = local_epochs
        self.num_clients = "N/A"
        self.server_clip_const = server_clip_const
        
        self.accuracy_metrics = {"train": [], "eval": [], "test": []} # Used to track model accuracies
        self.checkpoint_train_acc = checkpoint_train_acc
        self.checkpoint_test_acc = checkpoint_test_acc
        self.empirical_eps_tracker = []

        self.logger = logger
        
        self.dp_level = dp_level
        self.task = task 

        self.base_plot_path = plot_path
        self.base_result_path = result_path
        self.grad_sample_module = grad_sample_module
        
        self.benchmark_times = benchmark_times if benchmark_times else []
        self.text_processor = TextProcessorSent140() if dataset == "sent140" else TextProcessorShakes()

    def reset(self):
        """ Resets attributes that track a canary attack
        
        """
        self.canary_healths = []
        self.canaries = []
        self.canary_dot_prods = {"with_canary": [], "without_canary": []}
        self.init_canary_dot_prods = {"with_canary": [], "without_canary": []}
        self.batch_clip_percs = []
        self.clip_rates = []
        self.num_tests = 0

    def _plot_canary_hist(self, canary_metrics, suffix=""):
        """ Plots canary histogram and associated attack metrics for a canary that is being analysed

        Args:
            canary_metrics (dict): Dict of canary metrics
            suffix (str, optional): Plot name suffix. Defaults to "".
        """
        if np.isnan(np.sum(canary_metrics["dot_prods_without_canary"])) or np.isnan(np.sum(canary_metrics["dot_prods_with_canary"])):
            self.logger.info("WARNING - Some dot products are NaN, these are being removed for plotting...")
            canary_metrics["dot_prods_without_canary"] = np.array(canary_metrics["dot_prods_without_canary"])[~np.isnan(canary_metrics["dot_prods_without_canary"])]
            canary_metrics["dot_prods_with_canary"] =  np.array(canary_metrics["dot_prods_with_canary"])[~np.isnan( canary_metrics["dot_prods_with_canary"])]

        if len(canary_metrics["dot_prods_without_canary"]) == 0 or len(canary_metrics["dot_prods_with_canary"]) == 0 :
            self.logger.info("Dot products were empty, likely all nans, optimisation has failed. Canary norm is likely 0...")
            return 
        
        bins = 25
        bins=None
        plt.hist(canary_metrics["dot_prods_without_canary"], bins=bins, label="Without canary (" + self.canary_design_type + "), m=" + str(round(canary_metrics["without_canary_mean"], 5)) + " std=" + str(round(canary_metrics["without_canary_sd"], 5)))
        plt.hist(canary_metrics["dot_prods_with_canary"], bins=bins, label="W/ canary (" + self.canary_design_type + ") m=" + str(round(canary_metrics["with_canary_mean"], 5)) + " std=" + str(round(canary_metrics["with_canary_sd"], 5)))
        plt.vlines(canary_metrics["mia_threshold"], ymin=0, ymax=10, color="red")

        plot_title = self.task + " " + self.dp_level +  " num_clients=" + str(self.num_clients) + " local_steps=" + str(self.local_updates) + " init=" + self.canary_init + "\n"
        plot_title += "Design: naive" if  self.canary_design_type == "naive" else f"Design: {self.canary_design_type } {self.canary_loss}"
        plot_title += f" Local Batch Size={self.local_batch_size} epoch={self.canary_insert_epoch}, round={self.canary_insert_global_round}"
        
        if len(self.accuracy_metrics["train"]) > 0 and len(self.accuracy_metrics["test"]) > 0:
            plot_title += f" (Train, Test): {round(self.accuracy_metrics['train'][-1],2)}, {round(self.accuracy_metrics['test'][-1],2)}"
        if self.canary_setup == "holdout" and self.canary_design_type != "naive":
            plot_title += f"\n Design Sample={self.canary_design_sample_size} Design Pool={self.canary_design_pool_size}"
            if self.canary_loss != "loss1":
                plot_title += f" Minibatch= {self.canary_design_minibatch_size}"
        if self.canary_setup == "exact" and self.canary_design_type != "naive":
            plot_title += "\n Canary Health (min, max, mean): {min}, {max}, {mean}".format(min=str(round(np.min(canary_metrics["canary_health_list"]), 4)), 
                                                                                        max=str(np.round(max(canary_metrics["canary_health_list"]),4)), mean=str(round(np.mean(canary_metrics["canary_health_list"]), 4)))
        else:
            plot_title += f"\n  Canary norm={round(canary_metrics['canary_norm'],3)} Canary Health: {round(canary_metrics['canary_health_list'][0],5)}"

        plot_title += f" (Acc, Max Acc, AUC): {round(canary_metrics['mia_acc'], 4)}, {round(canary_metrics['mia_max_acc'],4)}, {round(canary_metrics['mia_auc'],4)}"
        plot_title += f"\n (eps, delta)=({round(canary_metrics['initial_epsilon'],4)}, {canary_metrics['initial_delta']}), sigma={round(canary_metrics['final_sigma'],4)}, empirical= {round(canary_metrics['empirical_eps'],4)}, ({round(canary_metrics['empirical_eps_lower'],4)}, {round(canary_metrics['empirical_eps_upper'],4)})"
        plt.title(plot_title, fontdict={"fontsize": 10})
        plt.ylabel("Freq")
        plt.xlabel(r'<S, grad(canary)>')
        plt.legend()
        plt.tight_layout()
        
        full_path = self.plot_path + suffix + ".png"
        plt.savefig(full_path, bbox_inches='tight')
        self.logger.info(f" Plot Saved: {full_path}")
        plt.clf()

    def _plot_canary_losses(self):
        """Plots the optimisation loss of an analysed canary.
        """
        smoothed_loss = np.mean(np.array(self.canary_losses)[:(len(self.canary_losses)//100)*100].reshape(-1,100), axis=1)
        data_list = [("canary_norms", self.canary_norms), ("canary_loss_full", self.canary_losses), 
                     ("canary_loss_last_epochs", self.canary_losses[-1000:]), ("canary_loss_smoothed", smoothed_loss)]
        
        for item in data_list:
            name, data = item
            plt.plot(range(0, len(data)), data)
            plt.title(name)
            plt.ylabel(name)
            plt.xlabel("Epoch")
            plt.tight_layout()
            full_path = self.plot_path + f"_{name}.png"
            plt.savefig(full_path)
            self.logger.info(f" Plot Saved: {full_path}")
            plt.clf()

    def _plot_pr_curve(self, precision, recall, auprc=0, suffix=""):
        """Plots pr curves of an analysed canary

        Args:
            precision (list): Precision values
            recall (list): Recall values
            auprc (int, optional): Optional AUPRC to display in the plot title. Defaults to 0.
            suffix (str, optional): Plot name suffix. Defaults to "".
        """
        for i in range(recall.shape[0]-1):
            plt.plot((recall[i],recall[i]),(precision[i],precision[i+1]),'b-') 
            plt.plot((recall[i],recall[i+1]),(precision[i+1],precision[i+1]),'b-') 

        plt.title(f"PR Curve - MAP={auprc}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plot_name = self.plot_path + "_pr_curve_" + suffix
        plt.savefig(plot_name)
        self.logger.info(f" Plot Saved: {plot_name}")
        plt.clf()

    def _save_results(self, canary_metrics, additional_args):
        """Checkpoint analysed canary attack

        Args:
            canary_metrics (dict): All canary metrics to checkpoint
            additional_args (dict): Additional args i.e, from a CanaryDesigner
        """
        all_args = canary_metrics
        all_args.update(self.__dict__)
        remove_list = ["grad_sample_module", "canaries", "logger", "canary_losses", "text_processor", "canary_dot_prods", "init_canary_dot_prods", "canary_norms"]
        for attr in remove_list:
            all_args.pop(attr)
            
        if additional_args is not None:
            all_args.update(vars(additional_args)) 

        experiment_dict = {}
        all_args["canary_health"] = all_args["canary_health_list"][0] if len(all_args["canary_health_list"]) == 1 else np.mean(all_args["canary_health_list"])
        columns = list(canary_metrics.keys())
        row = [all_args[col] for col in columns]

        experiment_dict["row"] = row
        experiment_dict["columns"] = columns
        
        torch.save(experiment_dict, self.result_path + ".tar")
        self.logger.info(f" Experiment metrics saved {self.result_path}")
        self.logger.info(f"Saved columns {columns}")
        self.logger.info(f"Canary insert epoch={all_args['canary_insert_epoch']}, global round={all_args['canary_insert_global_round']}")
        
    def _save_canary(self, batched_canary, title):
        """Saves an output of the designed canary. Either as an image of a .txt for NLP

        Args:
            batched_canary: Batch with a single canary
            title: Title of the canary output file
        """
        if self.canary_type == "image":
            if self.dataset == "femnist":
                plt.imshow(np.transpose(batched_canary[0]).numpy(), cmap="gray")
            else:
                plt.imshow(np.transpose(batched_canary[0].numpy(), (1, 2, 0)))

            plt.title(title)
            plt.axis("off")
            plt.savefig(self.plot_path + "_" + title + ".png")
            plt.clf()
        elif self.canary_type == "nlp":
            try:
                with open(self.plot_path + "_" + title + ".txt", 'w') as f:
                    f.write(self.text_processor.index_sequence_to_text(batched_canary[0]))
            except:
                plt.clf()
                self.logger.info("Saving nlp error...")

    def ci_eps(self, fp, fn, n_pos, n_neg, delta=1e-5, bound="lower"):
        """Calculate the 95% CI for empirial epsilon via the Clopper-Pearson method

        Args:
            fp (_type_): False positives
            fn (function): False negatives
            n_pos (_type_): Number of positive examples
            n_neg (_type_): Number of negative examples
            delta (_type_, optional): DP delta. Defaults to 10e-5.
            bound (str, optional): "upper" or "lower" CI bounds. Defaults to "lower".

        Returns:
            empirial eps
        """
        fp = int(fp)
        fn = int(fn)
        
        fp_result = binomtest(k=fp, n=n_pos)
        fn_result = binomtest(k=fn, n=n_neg)
        
        if bound == "lower":
            fp_hi = fp_result.proportion_ci().high
            fn_hi = fn_result.proportion_ci().high
        else:
            fp_hi = fp_result.proportion_ci().low
            fn_hi = fn_result.proportion_ci().low
            
        return self.empirical_eps(1-fn_hi,fp_hi, delta=delta, type=bound)

    def empirical_eps(self, tpr, fpr, delta=1e-5, type=""):
        """Calculate empirical epsilon

        Args:
            tpr: True Positive Rate (TPR)
            fpr: False Positive Rate (FPR)
            delta: DP delta. Defaults to 10e-5.
            type (str, optional): "lower" or "upper" for CI calculations. Defaults to "".

        Returns:
            empirical eps
        """
        x = []        
        if 1-tpr > 0:
            x.append((1-delta-fpr)/(1-tpr))
        if fpr > 0:
            x.append((1-delta-(1-tpr))/fpr)

        if len(x) <= 1 or max(x) < 0:
            print(f"Warning empirical eps=inf, type={type} - {fpr}, {1-tpr}")
            x = [float("inf")]

        return math.log(max(x))
    
    def _compute_empirical_eps(self, attack_results: AttackResults, use_max_acc_threshold=False):
        n_pos, n_neg = len(attack_results.scores_train), len(attack_results.scores_test)     
        delta = 1/(n_pos + n_neg)
       
        max_empirical_eps = 0
        _, scores = attack_results._get_scores_and_labels_ordered()
        tpr_fpr = attack_results.get_tpr_fpr()
        
        if use_max_acc_threshold: # Calculate empirical eps from max acc threshold
            max_acc_thresh = attack_results.get_max_accuracy_threshold()[0]
            tp = int((attack_results.scores_train >= max_acc_thresh).sum().item())
            fp = int((attack_results.scores_test >= max_acc_thresh).sum().item())
            max_fp, max_fn, max_tp, max_tn = fp, n_pos-tp, tp, n_neg-fp
            max_tpr, max_fpr = max_tp / (max_tp + max_fn), max_fp/(max_fp+max_tn)
            max_empirical_eps = self.empirical_eps(max_tpr, max_fpr, delta=delta)
        else: # Maximise empirical eps over TPR/FPR
            for i, t in enumerate(scores):
                tpr, fpr = tpr_fpr[0][i], tpr_fpr[1][i]
                empirical_eps = self.empirical_eps(tpr, fpr, delta=delta)
                acc = attack_results.get_accuracy(t)
                
                if empirical_eps > max_empirical_eps and (empirical_eps != float("inf") or acc == 1):
                    tp = int((attack_results.scores_train >= t).sum().item())
                    fp = int((attack_results.scores_test >= t).sum().item())
                    max_empirical_eps = empirical_eps
                    max_fp, max_fn, max_tp, max_tn = fp, n_pos-tp, tp, n_neg-fp
                    max_tpr, max_fpr = tpr, fpr

        empirical_eps_lower = self.ci_eps(max_fp, max_fn, n_pos=n_pos, n_neg=n_neg, delta=delta)
        empirical_eps_upper = self.ci_eps(max_fp, max_fn, bound="upper", n_pos=n_pos, n_neg=n_neg, delta=delta)
        return max_empirical_eps, empirical_eps_lower, empirical_eps_upper, max_fp, max_fn, max_tp, max_tn
    
    def _compute_canary_metrics(self, initial_privacy_budget, final_privacy_budget, type="canary", correct_bias=False, plot_prc=True, **kwargs): 
        """Computes canary and attack metrics for checkpointing

        Args:
            initial_privacy_budget (dict): Initial privacy budget of the model
            final_privacy_budget (dict): Final privacy budget at the attack round
            type (str, optional): Type of canary metrics, either "init" or "canary". Defaults to "canary".
            correct_bias (bool, optional): Debugging, if True computes corrected bias metrics. Defaults to False.
            plot_prc (bool, optional): If True will plot PR curves. Defaults to True.

        Returns:
            canary_metrics: dict of canary metrics to checkpoint
        """
        canary_metrics = {}
        canary_metrics.update(kwargs)
        bias = self.canary_design_bias if correct_bias else 0

        canary_metrics["with_canary_mean"] = np.round(np.mean(canary_metrics["dot_prods_with_canary"], axis=0)+bias,10)
        canary_metrics["with_canary_var"] = np.round(np.var(canary_metrics["dot_prods_with_canary"], axis=0),10)
        canary_metrics["without_canary_mean"] = np.round(np.mean(canary_metrics["dot_prods_without_canary"], axis=0)+bias,10)
        canary_metrics["without_canary_var"] = np.round(np.var(canary_metrics["dot_prods_without_canary"], axis=0),10)

        results = AttackResults(torch.tensor(canary_metrics["dot_prods_with_canary"])+bias, torch.tensor(canary_metrics["dot_prods_without_canary"])+bias)
        max_accuracy_threshold, max_accuracy = results.get_max_accuracy_threshold()
        tpr, fpr = results.get_tpr_fpr()
        precision, recall = results.get_precision_recall()
        auprc = results.get_map()

        canary_metrics["mia_auc"] = results.get_auc()
        canary_metrics["mia_threshold"] = max_accuracy_threshold
        canary_metrics["mia_max_acc"] = max_accuracy 
        canary_metrics["mia_acc"] = results.get_accuracy(threshold=0.5).item()

        if plot_prc:
            self._plot_pr_curve(precision, recall, auprc=auprc, suffix=type)

        n_pos = len(results.scores_test)
        n_neg = len(results.scores_train)
        n = n_pos + n_neg
        self.logger.info(f"=== Computing metrics for type={type}")
        self.logger.info(f"Number of tests={self.num_tests}, without={len(results.scores_train)}, with={len(results.scores_test)}, n={n}")
        
        empirical_eps, empirical_eps_lower, empirical_eps_upper, fp, fn, tp, tn = self._compute_empirical_eps(attack_results=results, use_max_acc_threshold=False)
        self.logger.info(f"n={n}, tp={tp}, fp={fp}, tn={tn}, fn={fn}")
        
        fpr = fp/(fp+tn) 
        fnr = fn/(fn+tp)
        tpr = 1-fnr
        self.logger.info(f"FPR={fpr}, TPR={tpr}, FNR={fnr}")
        self.logger.info(f"Type={type}, Acc= {canary_metrics['mia_acc']}, empirical eps={empirical_eps}, lower, upper =({empirical_eps_lower},{empirical_eps_upper})\n")
            
        canary_metrics["fp"] = fp
        canary_metrics["fn"] = fn
        canary_metrics["tp"] = tp
        canary_metrics["tn"] = tn
        canary_metrics["empirical_eps_lower"] = empirical_eps_lower
        canary_metrics["empirical_eps_upper"] = empirical_eps_upper
        canary_metrics["empirical_eps"] = empirical_eps
        canary_metrics["without_canary_sd"] = math.sqrt(canary_metrics["without_canary_var"])
        canary_metrics["with_canary_sd"] = math.sqrt(canary_metrics["with_canary_var"])
        canary_metrics["sd_gap"] = abs(canary_metrics["without_canary_sd"] - canary_metrics["with_canary_sd"])
        canary_metrics["loss_gap"] = np.min(canary_metrics["dot_prods_with_canary"])+bias - np.max(canary_metrics["dot_prods_without_canary"])+bias
        canary_metrics["batch_clip_percs"] = kwargs["batch_clip_percs"]

        if type == 'canary':
            self.empirical_eps_tracker.append((canary_metrics["empirical_eps_lower"],  canary_metrics["empirical_eps"],  canary_metrics["empirical_eps_upper"]))

        self._add_privacy_metrics(canary_metrics, initial_privacy_budget, type="initial")
        self._add_privacy_metrics(canary_metrics, final_privacy_budget, type="final")

        return canary_metrics 

    def _add_privacy_metrics(self, metrics, privacy_budget, type="final"):
        """Adds privacy budget to canary metrics 

        Args:
            metrics (Canary metrics): Canary metrics
            privacy_budget (dict): Privacy budget
            type (str, optional): Type. Defaults to "final".
        """
        metrics[f"{type}_epsilon"] = privacy_budget["epsilon"]
        metrics[f"{type}_delta"] = privacy_budget["delta"]
        metrics[f"{type}_sigma"] = privacy_budget["sigma"]
        
    def add_clip_rate(self, clip_rate):
        """Add a clip rate e.g. a % of model updates that were clipped in the current test round

        Args:
            clip_rate (float): clip percentage
        """
        self.clip_rates.append(clip_rate)

    def add_canary(self, canary):
        """Add a canary to be analysed

        Args:
            canary (Canary): canary
        """
        self.canaries.append(canary)

    def set_canary(self, canary):
        """Set a canary, replacing all old canaries being tracked

        Args:
            canary
        """
        self.canaries = [canary]

    def reset_canaries(self):
        """Reset all tracked canaries
        """
        self.canaries = []
        
    def set_grad_sample_module(self, model):
        """Set GradSampleModule, not used in FLSim

        Args:
            model (GSM)
        """
        self.grad_sample_module = GradSampleModule(copy.deepcopy(model))

    def set_accuracy_metrics(self, accuracy_metrics):
        """Set accuracy metrics of model to checkpoint in canary_metrics

        Args:
            accuracy_metrics: FLSim accuracy metrics
        """
        self.accuracy_metrics = accuracy_metrics
        self.current_train_acc = accuracy_metrics["train"][-1] if len(accuracy_metrics["train"]) > 0 else 0
        self.current_test_acc = accuracy_metrics["test"][-1] if len(accuracy_metrics["test"]) > 0 else 0

    def test_against_batch(self, criterion, batch, canary, device="cpu"):
        """Debugging only, not used in FLSim.

        Args:
            criterion: torch criterion
            batch: Batch to test canary presence
            canary (Canary): canary
            device (str, optional): torch device. Defaults to "cpu".
        """
        assert self.grad_sample_module is not None, "Must set_grad_sample_module() before testing with a batch"
        
        if canary not in self.canaries:
            self.add_canary(canary)

        # Compute required gradients
        batch_sample_grads, clip_count = compute_sample_grads(self.grad_sample_module, criterion, batch, device=device, clipping_const=self.canary_clip_const)

        clip_perc = round(clip_count / self.local_batch_size, 8)*100
        self.batch_clip_percs.append(clip_perc)
        self.logger.info(f" Clip Percentage {clip_perc}")

        aggregated_batch_grad = torch.sum(batch_sample_grads, axis=0)

        canary_norm = torch.norm(canary.grad).item()
        self.logger.info(f" Canary grad norm: {canary_norm}\n")
    
        self.canary_dot_prods["without_canary"].append(torch.dot(canary.grad, aggregated_batch_grad).item())
        self.init_canary_dot_prods["without_canary"].append(torch.dot(canary.init_grad, aggregated_batch_grad).item())

        self.canary_dot_prods["with_canary"].append(torch.dot(canary.grad, aggregated_batch_grad + canary.grad).item())
        self.init_canary_dot_prods["with_canary"].append(torch.dot(canary.init_grad, aggregated_batch_grad+canary.init_grad).item())

        self.num_tests += 1

    def test_against_agg_grad(self, canary, aggregated_model, lr, num_clients, clip_factor=1, type="with"):
        """Tests canary against aggregated model udpates by computing a dot-product score.

        Args:
            canary (Canary): canary to test
            aggregated_model (tensor): Aggregated clipped (noisy) model updates
            lr: Client lr
            num_clients: Number of clients in the current round
            clip_factor (int, optional): Clip factor (not used). Defaults to 1.
            type (str, optional): Type of the test, either "with" or "without" canary. Defaults to "with".
        """
        self.num_clients = num_clients
        aggregated_batch_grad = torch.tensor([])
        for p in aggregated_model.parameters():
            aggregated_batch_grad = torch.cat([aggregated_batch_grad, p.detach().clone().cpu().flatten()])
        aggregated_batch_grad =  num_clients * aggregated_batch_grad * 1/clip_factor
        
        self.logger.debug(f"Aggregated grads {aggregated_batch_grad}")
        self.logger.debug(f"Norm of aggregated grads {torch.norm(aggregated_batch_grad)}")
        self.logger.debug(f"Clip factor {clip_factor}, num clients {num_clients}, lr {lr}, Batch size {self.local_batch_size}")
        self.logger.debug(f"Aggregated scaled grads {aggregated_batch_grad}")
        self.logger.info(f"Canary grad norm {torch.norm(canary.grad)}, Canary clip const {self.canary_clip_const}")
        
        # if self.canary_design_type == "sample_grads": # Designing against unclipped updates (must unscale)
        # aggregated_batch_grad = (1/lr) * num_clients * self.local_batch_size * aggregated_batch_grad * 1/clip_factor
        # else: # Designing against clipped and scaled updates (so no need to unscale)

        # Aggregate attack scores
        if type == "with" or type == "without":
            if self.canary_design_reverse_server_clip: 
                canary_dot_prod = torch.dot(canary.grad/torch.norm(canary.grad)**2, aggregated_batch_grad).item()
            else: 
                if self.scale_canary_test and torch.norm(canary.grad) < self.server_clip_const:
                    canary_dot_prod = torch.dot(canary.grad/(torch.norm(canary.grad)**2)*self.server_clip_const, aggregated_batch_grad).item()
                else:
                    canary_dot_prod = torch.dot((canary.grad/torch.norm(canary.grad)*self.server_clip_const)/self.server_clip_const**2, aggregated_batch_grad).item()
            self.canary_dot_prods[type+"_canary"].append(canary_dot_prod)
        
        # Aggregate canary init scores
        init_dot_prod = torch.dot(canary.init_grad/torch.norm(canary.init_grad), aggregated_batch_grad).item()
        if type == "without":
            self.init_canary_dot_prods[type+"_canary"].append(init_dot_prod)
        elif type == "with_init":
            self.init_canary_dot_prods["with_canary"].append(init_dot_prod)

        self.num_tests += 1

    def analyse(self, global_round=0, initial_privacy_budget=None, final_privacy_budget=None, one_step_budget=None,
                    disable_init_metrics=False, disable_bias_metrics=False, 
                    plot_hists=True, plot_canaries=True, plot_losses=True, plot_prc=True, args=None):
        """Analyse current set canary and checkpoint associated attack metrics and plots

        Args:
            global_round (int, optional): Global FL round of the attack. Defaults to 0.
            initial_privacy_budget (dict, optional): Initial model privacy budget. Defaults to None.
            final_privacy_budget (dict, optional): Current model privacy budget. Defaults to None.
            one_step_budget (dict, optional): Model one-step budget. Defaults to None.
            disable_init_metrics (bool, optional): If True will not compute canary init metrics. Defaults to False.
            disable_bias_metrics (bool, optional): If False will not compute bias corrected metrics. Defaults to False.
            plot_hists (bool, optional): If False will not plot attack histograms. Defaults to True.
            plot_canaries (bool, optional): If False will not output canaries. Defaults to True.
            plot_losses (bool, optional): If False will not output canary optimisation loss plots. Defaults to True.
            plot_prc (bool, optional): If False will not plot PR curves. Defaults to True.
            args (dict, optional): Additional args for checkpointing. Defaults to None.
        """
        assert len(self.canaries) > 0, "Cannot anaylse() before test_against_agg_grad() or test_against_batch() at least once"
        
        if final_privacy_budget is None:
            final_privacy_budget = {"epsilon": float('inf'), "delta": 0, "sigma": 0}
        if initial_privacy_budget is None:
            initial_privacy_budget = {"epsilon": 0, "delta": 0, "sigma": 0}
        if one_step_budget is None:
            one_step_budget = {"epsilon": 0, "delta": 0, "sigma": 0}

        self.global_round = global_round
        self.plot_path = self.base_plot_path + f"_global_round={global_round}"
        self.result_path = self.base_result_path + f"_global_round={global_round}"

        self.canary_healths = [canary.health for canary in self.canaries]
        canary_norm = np.mean([torch.norm(canary.grad).item() for canary in self.canaries])
        init_norm = np.mean([torch.norm(canary.init_grad).item() for canary in self.canaries])
        final_loss = np.mean([canary.final_loss for canary in self.canaries])
        init_loss = np.mean([canary.init_loss for canary in self.canaries])
        
        # Save initial and final canaries
        if plot_canaries:
            self._save_canary(self.canaries[0].data, "final_canary class " + str(self.canaries[0].class_label))
            self._save_canary(self.canaries[0].init_data, "init_canary class " + str(self.canaries[0].class_label))

        # Compute and save canary metrics
        canary_metrics = self._compute_canary_metrics(initial_privacy_budget, final_privacy_budget, type="canary", plot_prc=plot_prc, dot_prods_with_canary=self.canary_dot_prods["with_canary"], dot_prods_without_canary=self.canary_dot_prods["without_canary"], 
                                                canary_norm=canary_norm, 
                                                canary_health_list=self.canary_healths, batch_clip_percs=self.batch_clip_percs, final_loss=final_loss)

        if not disable_bias_metrics:
            canary_bias_corrected_metrics = self._compute_canary_metrics(initial_privacy_budget, final_privacy_budget, type="bias_canary", plot_prc=False, correct_bias=True, dot_prods_with_canary=self.canary_dot_prods["with_canary"], dot_prods_without_canary=self.canary_dot_prods["without_canary"], 
                                                    canary_norm=canary_norm, 
                                                    canary_health_list=self.canary_healths, batch_clip_percs=self.batch_clip_percs, final_loss=final_loss)

            canary_bias_corrected_metrics["mia_threshold"] = 0.5
            canary_metrics["bias_corrected_acc"] = canary_bias_corrected_metrics["mia_acc"]
        
        if not disable_init_metrics:
            init_canary_metrics = self._compute_canary_metrics(initial_privacy_budget, final_privacy_budget, type="init", plot_prc=plot_prc, dot_prods_with_canary=self.init_canary_dot_prods["with_canary"], dot_prods_without_canary=self.init_canary_dot_prods["without_canary"], 
                                                canary_norm=init_norm, 
                                                canary_health_list=[0], batch_clip_percs=self.batch_clip_percs, final_loss=init_loss)

        canary_metrics["sd_improvement"] = "n/a" if disable_init_metrics else init_canary_metrics["without_canary_sd"] - canary_metrics["without_canary_sd"]
        canary_metrics["init_without_canary_sd"] =  "n/a" if disable_init_metrics else init_canary_metrics["without_canary_sd"]
        canary_metrics["init_with_canary_sd"] =  "n/a" if disable_init_metrics else init_canary_metrics["with_canary_sd"]
        canary_metrics["mia_acc_improvement"] =  "n/a" if disable_init_metrics else canary_metrics["mia_max_acc"] - init_canary_metrics["mia_max_acc"]
        canary_metrics["dot_prods_with_init_canary"] =  "n/a" if disable_init_metrics else self.init_canary_dot_prods["with_canary"]
        canary_metrics["dot_prods_without_init_canary"] =  "n/a" if disable_init_metrics else self.init_canary_dot_prods["without_canary"]
        canary_metrics["one_step_eps"] = one_step_budget["epsilon"]

        self.logger.info(f"One step privacy metrics (no sampling) (eps,delta)={one_step_budget['epsilon']}, {one_step_budget['delta']}, sigma={one_step_budget['sigma']}")
        self.logger.info(f"Initial privacy metrics (eps,delta)={canary_metrics['initial_epsilon']}, {canary_metrics['initial_delta']}, sigma={canary_metrics['initial_sigma']}")
        self.logger.info(f"Final privacy metrics (eps,delta)={canary_metrics['final_epsilon']}, {canary_metrics['final_delta']}, sigma={canary_metrics['final_sigma']}, sample rate={self.sample_rate}")
        self.logger.info(f"Empirical epsilon tracker {self.empirical_eps_tracker}\n")
        self.logger.info(f"Checkpoint train acc {self.checkpoint_train_acc} Checkpoint test acc {self.checkpoint_test_acc}")
        self.logger.info(f"Current train acc {self.current_train_acc} Current test acc {self.current_test_acc}")
        self.logger.info(f"All accuracy metrics {self.accuracy_metrics}\n")
        if not disable_init_metrics:
            self.logger.info(f" SD Improvement: {canary_metrics['sd_improvement']}")
            self.logger.info(f" MIA Acc Improvement: {canary_metrics['mia_acc_improvement']}\n")

        # Save canary metrics
        self._save_results(canary_metrics, args)
        # Plot and save dot product histograms
        if plot_hists:
            self._plot_canary_hist(canary_metrics, suffix="_canary") # Final canary
            if not disable_bias_metrics:
                self._plot_canary_hist(canary_bias_corrected_metrics, suffix="_bias_corrected_canary") # Final canary (bias-corrected)
            if not disable_init_metrics:
                self._plot_canary_hist(init_canary_metrics, suffix="_init") # Initial canary
        # Save canary optim losses
        if plot_losses:
            self._plot_canary_losses()
            
        self.logger.info(f"Minibatch, sample size, pool size, {self.canary_design_minibatch_size, self.canary_design_sample_size, self.canary_design_pool_size}")
        self.logger.info(f"Actual minibatch, sample size, pool size, {self.actual_minibatch_size, self.actual_sample_size, self.actual_pool_size}")