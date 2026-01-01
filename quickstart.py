#!/usr/bin/env python3
"""
Quick start script for healthcare fraud detection.

This script provides a simple way to get started with the fraud detection system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import FraudDataset
from models import create_model
from metrics import FraudDetectionMetrics
from utils import set_deterministic_seed, setup_logging

def main():
    """Quick start demonstration."""
    
    print("🏥 Healthcare Fraud Detection - Quick Start")
    print("=" * 50)
    print("⚠️  RESEARCH DEMONSTRATION ONLY - NOT FOR CLINICAL USE")
    print()
    
    # Set up
    set_deterministic_seed(42)
    logger = setup_logging("INFO")
    
    # Generate synthetic data
    print("Generating synthetic healthcare claims data...")
    dataset = FraudDataset(n_samples=2000)
    df = dataset.data
    
    print(f"✓ Generated {len(df)} samples with {df['fraud_label'].mean():.1%} fraud rate")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_train_test_split(
        test_size=0.2, val_size=0.1, random_state=42, stratify=True
    )
    
    print(f"✓ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model = create_model("xgboost", n_estimators=50, max_depth=6)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    print("✓ Model training completed")
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics_calculator = FraudDetectionMetrics(threshold=0.5)
    metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred, y_proba)
    
    # Display results
    print("\n📊 Model Performance:")
    print("-" * 30)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1_score']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.3f}")
    
    # Generate report
    report = metrics_calculator.generate_report(y_test, y_pred, y_proba)
    print(f"\n📋 Detailed Report:")
    print("-" * 30)
    print(report)
    
    print("\n✅ Quick start completed successfully!")
    print("\nNext steps:")
    print("- Run 'python scripts/train.py --help' for training options")
    print("- Run 'streamlit run demo/app.py' for interactive demo")
    print("- Check notebooks/ for detailed examples")
    print("\n⚠️  Remember: This is for research only, not clinical use!")

if __name__ == "__main__":
    main()
