"""
Evaluation metrics for healthcare fraud detection.

This module provides comprehensive evaluation metrics including clinical metrics,
calibration analysis, and fairness assessment for fraud detection models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class FraudDetectionMetrics:
    """Comprehensive metrics for fraud detection evaluation."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize metrics calculator.
        
        Args:
            threshold: Classification threshold
        """
        self.threshold = threshold
        self.metrics = {}
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_classification_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))
            metrics.update(self._calculate_calibration_metrics(y_true, y_proba))
        
        # Clinical metrics
        metrics.update(self._calculate_clinical_metrics(y_true, y_pred, y_proba))
        
        self.metrics = metrics
        return metrics
    
    def _calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        metrics = {}
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        
        # Clinical metrics
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Additional metrics
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def _calculate_probability_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate probability-based metrics."""
        metrics = {}
        
        # Handle binary classification
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_binary = y_proba[:, 1]  # Use positive class probability
        else:
            y_proba_binary = y_proba.flatten()
        
        # AUC metrics
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba_binary)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba_binary)
        
        # Log loss
        metrics["log_loss"] = log_loss(y_true, y_proba_binary)
        
        # Brier score
        metrics["brier_score"] = brier_score_loss(y_true, y_proba_binary)
        
        return metrics
    
    def _calculate_calibration_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate calibration metrics."""
        metrics = {}
        
        # Handle binary classification
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_binary = y_proba[:, 1]
        else:
            y_proba_binary = y_proba.flatten()
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_proba_binary)
        metrics["expected_calibration_error"] = ece
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, y_proba_binary)
        metrics["max_calibration_error"] = mce
        
        return metrics
    
    def _calculate_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_clinical_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate clinical-specific metrics."""
        metrics = {}
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Clinical utility metrics
        metrics["positive_likelihood_ratio"] = (
            tp / (tp + fn) / (fp / (fp + tn)) if (fp + tn) > 0 else 0
        )
        metrics["negative_likelihood_ratio"] = (
            fn / (fn + tp) / (tn / (tn + fp)) if (tn + fp) > 0 else 0
        )
        
        # Diagnostic odds ratio
        if tn > 0 and fp > 0:
            metrics["diagnostic_odds_ratio"] = (tp * tn) / (fp * fn)
        else:
            metrics["diagnostic_odds_ratio"] = 0
        
        # Youden's J statistic
        metrics["youden_j"] = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
        
        return metrics
    
    def calculate_threshold_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray, thresholds: Optional[List[float]] = None
    ) -> Dict[str, List[float]]:
        """Calculate metrics at different thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        # Handle binary classification
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_binary = y_proba[:, 1]
        else:
            y_proba_binary = y_proba.flatten()
        
        metrics = {
            "threshold": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "specificity": [],
            "sensitivity": [],
            "accuracy": []
        }
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba_binary >= threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            metrics["threshold"].append(threshold)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1_score"].append(f1)
            metrics["specificity"].append(specificity)
            metrics["sensitivity"].append(sensitivity)
            metrics["accuracy"].append(accuracy)
        
        return metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curve."""
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_binary = y_proba[:, 1]
        else:
            y_proba_binary = y_proba.flatten()
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_binary)
        auc_score = roc_auc_score(y_true, y_proba_binary)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
        """Plot Precision-Recall curve."""
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_binary = y_proba[:, 1]
        else:
            y_proba_binary = y_proba.flatten()
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_binary)
        pr_auc = average_precision_score(y_true, y_proba_binary)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
        """Plot calibration curve."""
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba_binary = y_proba[:, 1]
        else:
            y_proba_binary = y_proba.flatten()
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba_binary, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'], 
                   yticklabels=['Normal', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Formatted report string
        """
        metrics = self.calculate_all_metrics(y_true, y_pred, y_proba)
        
        report = "Healthcare Fraud Detection Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Classification metrics
        report += "Classification Metrics:\n"
        report += f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n"
        report += f"  Precision: {metrics.get('precision', 0):.4f}\n"
        report += f"  Recall (Sensitivity): {metrics.get('recall', 0):.4f}\n"
        report += f"  Specificity: {metrics.get('specificity', 0):.4f}\n"
        report += f"  F1-Score: {metrics.get('f1_score', 0):.4f}\n"
        report += f"  PPV: {metrics.get('ppv', 0):.4f}\n"
        report += f"  NPV: {metrics.get('npv', 0):.4f}\n\n"
        
        # Probability metrics
        if y_proba is not None:
            report += "Probability Metrics:\n"
            report += f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}\n"
            report += f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}\n"
            report += f"  Log Loss: {metrics.get('log_loss', 0):.4f}\n"
            report += f"  Brier Score: {metrics.get('brier_score', 0):.4f}\n\n"
            
            # Calibration metrics
            report += "Calibration Metrics:\n"
            report += f"  Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}\n"
            report += f"  Max Calibration Error: {metrics.get('max_calibration_error', 0):.4f}\n\n"
        
        # Clinical metrics
        report += "Clinical Metrics:\n"
        report += f"  Positive Likelihood Ratio: {metrics.get('positive_likelihood_ratio', 0):.4f}\n"
        report += f"  Negative Likelihood Ratio: {metrics.get('negative_likelihood_ratio', 0):.4f}\n"
        report += f"  Diagnostic Odds Ratio: {metrics.get('diagnostic_odds_ratio', 0):.4f}\n"
        report += f"  Youden's J Statistic: {metrics.get('youden_j', 0):.4f}\n"
        
        return report


class FairnessMetrics:
    """Fairness evaluation metrics for fraud detection."""
    
    def __init__(self, sensitive_attribute: str):
        """Initialize fairness metrics.
        
        Args:
            sensitive_attribute: Name of sensitive attribute column
        """
        self.sensitive_attribute = sensitive_attribute
    
    def calculate_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_groups: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate fairness metrics across sensitive groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_groups: Sensitive group assignments
            
        Returns:
            Dictionary of fairness metrics by group
        """
        unique_groups = np.unique(sensitive_groups)
        metrics_by_group = {}
        
        for group in unique_groups:
            group_mask = sensitive_groups == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            # Calculate metrics for this group
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            
            metrics = {
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
                "group_size": len(y_true_group),
                "fraud_rate": y_true_group.mean()
            }
            
            metrics_by_group[str(group)] = metrics
        
        return metrics_by_group
    
    def calculate_disparity_metrics(self, metrics_by_group: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate disparity metrics across groups.
        
        Args:
            metrics_by_group: Metrics calculated for each group
            
        Returns:
            Dictionary of disparity metrics
        """
        groups = list(metrics_by_group.keys())
        if len(groups) < 2:
            return {}
        
        disparities = {}
        
        for metric in ["accuracy", "precision", "recall", "specificity", "fpr", "fnr"]:
            values = [metrics_by_group[group][metric] for group in groups]
            disparities[f"{metric}_max_diff"] = max(values) - min(values)
            disparities[f"{metric}_std"] = np.std(values)
        
        return disparities
