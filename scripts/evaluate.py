#!/usr/bin/env python3
"""
Evaluation script for healthcare fraud detection models.

This script provides comprehensive evaluation of trained fraud detection models
including metrics calculation, visualization, and comparison across different models.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import FraudDataset
from models import BaseFraudDetector
from metrics import FraudDetectionMetrics, FairnessMetrics
from utils import setup_logging, ConfigManager, suppress_warnings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate healthcare fraud detection models")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to test data file (if not provided, synthetic data will be generated)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="assets/plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare multiple models if multiple model paths provided"
    )
    
    return parser.parse_args()


def load_model(model_path: str) -> BaseFraudDetector:
    """Load trained model from file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = BaseFraudDetector.load(model_path)
    
    return model


def load_test_data(
    config: DictConfig,
    data_path: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load test data.
    
    Args:
        config: Configuration object
        data_path: Path to data file
        
    Returns:
        Tuple of (X_test, y_test)
    """
    logger = logging.getLogger(__name__)
    
    if data_path:
        logger.info(f"Loading test data from {data_path}")
        dataset = FraudDataset(
            data_path=data_path,
            target_column=config.data.target_column,
            feature_columns=config.data.feature_columns,
            deidentify=True
        )
    else:
        logger.info("Generating synthetic test data")
        dataset = FraudDataset(
            target_column=config.data.target_column,
            feature_columns=config.data.feature_columns,
            deidentify=True
        )
    
    # Get test split
    _, _, X_test, _, _, y_test = dataset.get_train_test_split(
        test_size=config.data.test_split,
        val_size=config.data.val_split,
        random_state=config.data.random_seed,
        stratify=True
    )
    
    logger.info(f"Test data loaded - {len(X_test)} samples")
    return X_test, y_test


def evaluate_single_model(
    model: BaseFraudDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: DictConfig,
    plots_dir: str,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate a single model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        config: Configuration object
        plots_dir: Directory to save plots
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating {model.model_name} model")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics_calculator = FraudDetectionMetrics(threshold=threshold)
    metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred, y_proba)
    
    # Generate plots
    logger.info("Generating evaluation plots")
    
    # ROC curve
    metrics_calculator.plot_roc_curve(
        y_test, y_proba,
        save_path=f"{plots_dir}/roc_curve_{model.model_name}.png"
    )
    
    # Precision-Recall curve
    metrics_calculator.plot_precision_recall_curve(
        y_test, y_proba,
        save_path=f"{plots_dir}/pr_curve_{model.model_name}.png"
    )
    
    # Calibration curve
    metrics_calculator.plot_calibration_curve(
        y_test, y_proba,
        save_path=f"{plots_dir}/calibration_curve_{model.model_name}.png"
    )
    
    # Confusion matrix
    metrics_calculator.plot_confusion_matrix(
        y_test, y_pred,
        save_path=f"{plots_dir}/confusion_matrix_{model.model_name}.png"
    )
    
    # Generate evaluation report
    report = metrics_calculator.generate_report(y_test, y_pred, y_proba)
    logger.info(f"\n{report}")
    
    return metrics


def compare_models(
    models: List[BaseFraudDetector],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    plots_dir: str
) -> pd.DataFrame:
    """Compare multiple models.
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save results
        plots_dir: Directory to save plots
        
    Returns:
        DataFrame with comparison results
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Comparing {len(models)} models")
    
    comparison_results = []
    
    for model in models:
        logger.info(f"Evaluating {model.model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics_calculator = FraudDetectionMetrics(threshold=0.5)
        metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred, y_proba)
        
        # Add model name
        metrics["model"] = model.model_name
        comparison_results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Sort by ROC-AUC (descending)
    if "roc_auc" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("roc_auc", ascending=False)
    
    # Save comparison results
    comparison_path = Path(output_dir) / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Create comparison plots
    create_comparison_plots(comparison_df, plots_dir)
    
    return comparison_df


def create_comparison_plots(comparison_df: pd.DataFrame, plots_dir: str) -> None:
    """Create comparison plots.
    
    Args:
        comparison_df: DataFrame with comparison results
        plots_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger = logging.getLogger(__name__)
    
    # Set style
    plt.style.use("seaborn-v0_8")
    
    # Metrics to plot
    metrics_to_plot = ["roc_auc", "pr_auc", "f1_score", "precision", "recall"]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in comparison_df.columns:
            ax = axes[i]
            bars = ax.bar(comparison_df["model"], comparison_df[metric])
            ax.set_title(f"{metric.upper()}")
            ax.set_ylabel("Score")
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Comparison plots created")


def save_evaluation_results(
    metrics: Dict[str, float],
    model_name: str,
    output_dir: str
) -> None:
    """Save evaluation results.
    
    Args:
        metrics: Evaluation metrics
        model_name: Name of the model
        output_dir: Directory to save results
    """
    logger = logging.getLogger(__name__)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = Path(output_dir) / f"{model_name}_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save detailed report
    report_path = Path(output_dir) / f"{model_name}_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Evaluation Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    logger.info(f"Report saved to {report_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, "logs/evaluation.log")
    
    # Suppress warnings
    suppress_warnings()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    logger.info(f"Starting model evaluation")
    logger.info(f"Model path: {args.model_path}")
    
    # Setup directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Load test data
        X_test, y_test = load_test_data(config, args.data_path)
        
        if args.compare_models:
            # Compare multiple models
            model_paths = args.model_path.split(",")
            models = [load_model(path.strip()) for path in model_paths]
            
            comparison_df = compare_models(
                models, X_test, y_test, args.output_dir, args.plots_dir
            )
            
            logger.info("\nModel Comparison Results:")
            logger.info(comparison_df.to_string(index=False))
            
        else:
            # Evaluate single model
            model = load_model(args.model_path)
            
            metrics = evaluate_single_model(
                model, X_test, y_test, config, args.plots_dir, args.threshold
            )
            
            # Save results
            save_evaluation_results(metrics, model.model_name, args.output_dir)
            
            logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
