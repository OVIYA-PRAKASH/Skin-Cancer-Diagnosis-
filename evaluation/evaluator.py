"""
Phase 12 & 13: Testing Pipeline and Metrics.
Implements the MultiModalEvaluator to execute post-training model evaluations on test datasets.
Generates macro-averaged F1, precision, recall, confusion matrix heatmaps, and multi-class One-vs-Rest ROC-AUC curves.
Saves publication-quality figures directly to outputs/.
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
# Use Agg backend to prevent Tkinter GUI errors in server/background contexts
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from typing import Any, Dict

logger = logging.getLogger("SkinCancerAI.Evaluator")


class MultiModalEvaluator:
    """
    Evaluation suite that executes deep tests on a model, calculates scientific
    metrics, and plots ROC curves and confusion matrices.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: list[str],
        device: torch.device,
        output_dir: str = "outputs"
    ):
        """
        Initializes the evaluator.

        Args:
            model (nn.Module): Pre-trained HybridMultiModalClassifier model.
            test_loader (DataLoader): PyTorch DataLoader for the test split.
            class_names (list[str]): Names of the target skin diseases (alphabetical).
            device (torch.device): Device context (cuda or cpu).
            output_dir (str): Folder where visual plots will be saved.
        """
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Evaluator initialized for target classes: {self.class_names}")

    def evaluate(self) -> dict[str, Any]:
        """
        Runs evaluation on the test set, compiling true labels, predictions, and probabilities.

        Returns:
            dict[str, Any]: Complete metrics dictionary.
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []

        logger.info("Executing test split inference pass...")
        with torch.no_grad():
            for images, metadata, labels in self.test_loader:
                images = images.to(self.device)
                metadata = metadata.to(self.device)

                # Forward pass
                # We use softmax to map output logits to probability scores
                logits = self.model(images, metadata)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Convert to numpy arrays for calculation
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # 1. Calculate General metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names)), zero_division=0
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        logger.info(f"Test Accuracy reached: {accuracy*100:.2f}%")
        logger.info(f"Test Macro-F1 reached: {macro_f1:.4f}")

        # 2. Calculate Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))

        # 3. Calculate multi-class ROC AUC
        roc_auc_scores = {}
        # Binarize labels for One-vs-Rest (OvR) evaluation
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        # If dataset contains only 1 class in subset (edge case), handle binarization dimensions
        if len(self.class_names) == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

        for idx, name in enumerate(self.class_names):
            if len(np.unique(y_true_bin[:, idx])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
                auc_val = float(auc(fpr, tpr))
            else:
                fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
                auc_val = np.nan
            
            roc_auc_scores[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": auc_val
            }

        # Calculate Macro ROC AUC
        macro_auc_list = [roc_auc_scores[name]["auc"] for name in self.class_names]
        # Calculate mean ignoring nan values for class categories with 0 occurrences in evaluation slice
        macro_auc = float(np.nanmean(macro_auc_list)) if not np.all(np.isnan(macro_auc_list)) else np.nan
        logger.info(f"Test Macro-AUC reached: {macro_auc:.4f}")

        # Store complete results
        results = {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "macro_auc": macro_auc,
            "per_class": {
                name: {
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                    "auc": roc_auc_scores[name]["auc"]
                } for idx, name in enumerate(self.class_names)
            },
            "confusion_matrix": cm.tolist(),
            "roc_auc_curves": roc_auc_scores
        }

        # 4. Plot visual outputs
        self.plot_confusion_matrix(cm)
        self.plot_roc_curves(y_true_bin, y_prob)
        self.print_clinical_report(results)

        return results

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Plots a high-resolution Confusion Matrix heatmap.

        Args:
            cm (np.ndarray): Confusion matrix array.
        """
        plt.figure(figsize=(8, 6), dpi=300)
        # Normalize the confusion matrix by rows (true targets) safely to prevent division-by-zero on mock sets
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_percent = np.where(row_sums > 0, cm.astype("float") / row_sums, 0.0)

        # Build labels containing both count and percentage
        labels = (np.array([f"{count}\n({percent*100:.1f}%)" 
                  for count, percent in zip(cm.flatten(), cm_percent.flatten())])).reshape(cm.shape)

        sns.heatmap(
            cm_percent,
            annot=labels,
            fmt="",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True
        )

        plt.title("Clinical Diagnostic Confusion Matrix (Normalized)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Predicted Diagnosis", fontsize=11, labelpad=10)
        plt.ylabel("True Diagnosis", fontsize=11, labelpad=10)
        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix plot saved to: {fig_path}")

    def plot_roc_curves(self, y_true_bin: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Plots One-vs-Rest (OvR) ROC curves for all classes.

        Args:
            y_true_bin (np.ndarray): Binarized true labels.
            y_prob (np.ndarray): Prediction probability scores.
        """
        plt.figure(figsize=(8, 6), dpi=300)
        
        # Color palette for the 7 disease classes
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

        for idx, name in enumerate(self.class_names):
            if len(np.unique(y_true_bin[:, idx])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
                roc_auc = auc(fpr, tpr)
                label_text = f"ROC {name} (AUC = {roc_auc:.3f})"
            else:
                fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
                label_text = f"ROC {name} (AUC = N/A)"
                
            plt.plot(
                fpr,
                tpr,
                color=colors[idx % len(colors)],
                lw=2,
                label=label_text
            )

        # Plot micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(
            fpr_micro,
            tpr_micro,
            color="black",
            linestyle=":",
            linewidth=2.5,
            label=f"Micro-average (AUC = {roc_auc_micro:.3f})"
        )

        # Plot baseline random classifier
        plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=10)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, labelpad=10)
        plt.title("Multi-class One-vs-Rest (OvR) ROC Curves", fontsize=14, fontweight="bold", pad=15)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, "roc_curves.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Multi-class ROC curves saved to: {fig_path}")

    def print_clinical_report(self, results: dict[str, Any]) -> None:
        """
        Logs a publication-ready clinical diagnostic evaluation report.

        Args:
            results (dict[str, Any]): Metrics dictionary.
        """
        report = []
        report.append("\n" + "="*60)
        report.append("          CLINICAL DIAGNOSTIC EVALUATION REPORT          ")
        report.append("="*60)
        report.append(f"Overall Accuracy:  {results['accuracy']*100:.2f}%")
        report.append(f"Macro Precision:   {results['macro_precision']:.4f}")
        report.append(f"Macro Recall:      {results['macro_recall']:.4f}")
        report.append(f"Macro F1-Score:    {results['macro_f1']:.4f}")
        report.append(f"Macro ROC-AUC:     {results['macro_auc']:.4f}")
        report.append("-"*60)
        report.append(f"{'Class (dx)':<12} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'AUC':<9} | {'Count':<5}")
        report.append("-"*60)
        
        for name in self.class_names:
            class_stats = results["per_class"][name]
            report.append(
                f"{name:<12} | "
                f"{class_stats['precision']:<9.4f} | "
                f"{class_stats['recall']:<9.4f} | "
                f"{class_stats['f1']:<9.4f} | "
                f"{class_stats['auc']:<9.4f} | "
                f"{class_stats['support']:<5}"
            )
        report.append("="*60 + "\n")
        
        # Log to system logger
        logger.info("\n".join(report))
