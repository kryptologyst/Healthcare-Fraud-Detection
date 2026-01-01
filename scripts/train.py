#!/usr/bin/env python3
"""
Training script for healthcare fraud detection models.

This script provides a comprehensive training pipeline for various fraud detection
models including traditional ML, gradient boosting, and deep learning approaches.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import FraudDataset, FeatureEngineer
from models import create_model, BaseFraudDetector
from metrics import FraudDetectionMetrics, FairnessMetrics
from utils import set_deterministic_seed, setup_logging, ConfigManager, suppress_warnings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train healthcare fraud detection models")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["isolation_forest", "xgboost", "lightgbm", "catboost", "tabnet"],
        default="xgboost",
        help="Type of model to train"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data file (if not provided, synthetic data will be generated)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="assets/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="assets/plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--deidentify",
        action="store_true",
        default=True,
        help="De-identify data (default: True)"
    )
    
    return parser.parse_args()


def setup_directories(output_dir: str, results_dir: str, plots_dir: str) -> None:
    """Create necessary directories."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def load_and_prepare_data(
    config: DictConfig,
    data_path: Optional[str] = None,
    deidentify: bool = True
) -> Tuple[FraudDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare data for training.
    
    Args:
        config: Configuration object
        data_path: Path to data file
        deidentify: Whether to de-identify data
        
    Returns:
        Tuple of (dataset, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger = logging.getLogger(__name__)
    
    # Load or generate dataset
    if data_path:
        logger.info(f"Loading data from {data_path}")
        dataset = FraudDataset(
            data_path=data_path,
            target_column=config.data.target_column,
            feature_columns=config.data.feature_columns,
            deidentify=deidentify
        )
    else:
        logger.info("Generating synthetic dataset")
        dataset = FraudDataset(
            target_column=config.data.target_column,
            feature_columns=config.data.feature_columns,
            deidentify=deidentify
        )
    
    # Apply feature engineering
    logger.info("Applying feature engineering")
    dataset.data = FeatureEngineer.create_interaction_features(dataset.data)
    dataset.data = FeatureEngineer.create_risk_scores(dataset.data)
    
    # Update feature columns after engineering
    dataset._prepare_features()
    
    # Get train/val/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_train_test_split(
        test_size=config.data.test_split,
        val_size=config.data.val_split,
        random_state=config.data.random_seed,
        stratify=True
    )
    
    logger.info(f"Data prepared - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Feature dimensions: {X_train.shape[1]}")
    
    return dataset, X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    model: BaseFraudDetector,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[DictConfig] = None
) -> BaseFraudDetector:
    """Train the fraud detection model.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration object
        
    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training {model.model_name} model")
    
    # Prepare training arguments
    train_kwargs = {}
    
    if model.model_name == "TabNet":
        # TabNet requires validation data for early stopping
        if X_val is not None and y_val is not None:
            train_kwargs.update({
                "X_val": X_val,
                "y_val": y_val,
                "epochs": config.training.epochs,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate
            })
    elif model.model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        # Gradient boosting models can use validation for early stopping
        if X_val is not None and y_val is not None:
            train_kwargs.update({
                "eval_set": [(X_val, y_val)],
                "early_stopping_rounds": config.training.early_stopping,
                "verbose": config.logging.level == "DEBUG"
            })
    
    # Train the model
    model.fit(X_train, y_train, **train_kwargs)
    
    logger.info(f"{model.model_name} model training completed")
    return model


def evaluate_model(
    model: BaseFraudDetector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: DictConfig,
    plots_dir: str
) -> Dict[str, float]:
    """Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        config: Configuration object
        plots_dir: Directory to save plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating {model.model_name} model")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics_calculator = FraudDetectionMetrics(threshold=config.evaluation.threshold)
    metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred, y_proba)
    
    # Generate plots if requested
    if config.evaluation.plot_curves:
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


def save_results(
    model: BaseFraudDetector,
    metrics: Dict[str, float],
    output_dir: str,
    results_dir: str
) -> None:
    """Save model and results.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        output_dir: Directory to save model
        results_dir: Directory to save results
    """
    logger = logging.getLogger(__name__)
    
    # Save model
    model_path = Path(output_dir) / f"{model.model_name}_model.pkl"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = Path(results_dir) / f"{model.model_name}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save evaluation report
    report_path = Path(results_dir) / f"{model.model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model.model_name}\n")
        f.write("=" * 50 + "\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    logger.info(f"Report saved to {report_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, "logs/training.log")
    
    # Suppress warnings
    suppress_warnings()
    
    # Set random seed
    set_deterministic_seed(args.seed)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Override config with command line arguments
    if args.seed != 42:
        config.data.random_seed = args.seed
    
    logger.info(f"Starting training with {args.model_type} model")
    logger.info(f"Configuration: {args.config}")
    
    # Setup directories
    setup_directories(args.output_dir, args.results_dir, args.plots_dir)
    
    try:
        # Load and prepare data
        dataset, X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
            config, args.data_path, args.deidentify
        )
        
        # Create model
        model_config = config_manager.get_model_config(args.model_type)
        
        # Set input dimension for TabNet
        if args.model_type == "tabnet":
            model_config["input_dim"] = X_train.shape[1]
        
        model = create_model(args.model_type, **model_config)
        
        # Train model
        trained_model = train_model(
            model, X_train, y_train, X_val, y_val, config
        )
        
        # Evaluate model
        metrics = evaluate_model(
            trained_model, X_test, y_test, config, args.plots_dir
        )
        
        # Save results
        save_results(trained_model, metrics, args.output_dir, args.results_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
