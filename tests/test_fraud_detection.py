"""
Test suite for healthcare fraud detection project.

This module contains unit tests for all major components of the fraud detection system.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import FraudDataset, FeatureEngineer
from models import create_model, IsolationForestDetector, XGBoostDetector, LightGBMDetector
from metrics import FraudDetectionMetrics, FairnessMetrics
from utils import set_deterministic_seed, get_device, deidentify_data


class TestFraudDataset(unittest.TestCase):
    """Test cases for FraudDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_deterministic_seed(42)
        self.dataset = FraudDataset(n_samples=1000)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        self.assertIsInstance(self.dataset.data, pd.DataFrame)
        self.assertGreater(len(self.dataset.data), 0)
        self.assertIn("fraud_label", self.dataset.data.columns)
    
    def test_feature_preparation(self):
        """Test feature preparation."""
        features = self.dataset.get_features()
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(features.shape[1], 0)
    
    def test_train_test_split(self):
        """Test train/test split."""
        X_train, X_val, X_test, y_train, y_val, y_test = self.dataset.get_train_test_split()
        
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertEqual(len(X_train), len(y_train))
        self.assertGreater(len(X_train), 0)
    
    def test_class_weights(self):
        """Test class weight calculation."""
        weights = self.dataset.get_class_weights()
        self.assertIsInstance(weights, dict)
        self.assertIn(0, weights)
        self.assertIn(1, weights)


class TestModels(unittest.TestCase):
    """Test cases for model classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_deterministic_seed(42)
        
        # Create synthetic data
        self.dataset = FraudDataset(n_samples=500)
        X_train, X_val, X_test, y_train, y_val, y_test = self.dataset.get_train_test_split()
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def test_isolation_forest(self):
        """Test Isolation Forest model."""
        model = create_model("isolation_forest")
        
        # Train
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Predict
        predictions = model.predict(self.X_test)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_xgboost(self):
        """Test XGBoost model."""
        model = create_model("xgboost", n_estimators=10)
        
        # Train
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Predict
        predictions = model.predict(self.X_test)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_lightgbm(self):
        """Test LightGBM model."""
        model = create_model("lightgbm", n_estimators=10)
        
        # Train
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
        
        # Predict
        predictions = model.predict(self.X_test)
        probabilities = model.predict_proba(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))


class TestMetrics(unittest.TestCase):
    """Test cases for metrics classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_deterministic_seed(42)
        
        # Create synthetic predictions
        self.y_true = np.random.binomial(1, 0.1, 100)
        self.y_pred = np.random.binomial(1, 0.1, 100)
        self.y_proba = np.random.random((100, 2))
        self.y_proba = self.y_proba / self.y_proba.sum(axis=1, keepdims=True)
    
    def test_fraud_detection_metrics(self):
        """Test FraudDetectionMetrics class."""
        metrics_calculator = FraudDetectionMetrics(threshold=0.5)
        metrics = metrics_calculator.calculate_all_metrics(self.y_true, self.y_pred, self.y_proba)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
    
    def test_threshold_metrics(self):
        """Test threshold-based metrics."""
        metrics_calculator = FraudDetectionMetrics(threshold=0.5)
        threshold_metrics = metrics_calculator.calculate_threshold_metrics(
            self.y_true, self.y_proba
        )
        
        self.assertIsInstance(threshold_metrics, dict)
        self.assertIn("threshold", threshold_metrics)
        self.assertIn("precision", threshold_metrics)
        self.assertIn("recall", threshold_metrics)
    
    def test_fairness_metrics(self):
        """Test FairnessMetrics class."""
        sensitive_groups = np.random.choice(["A", "B"], 100)
        
        fairness_calculator = FairnessMetrics("group")
        fairness_metrics = fairness_calculator.calculate_fairness_metrics(
            self.y_true, self.y_pred, sensitive_groups
        )
        
        self.assertIsInstance(fairness_metrics, dict)
        self.assertGreater(len(fairness_metrics), 0)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Generate random numbers
        np_random = np.random.random(10)
        torch_random = None
        
        try:
            import torch
            torch_random = torch.rand(10).numpy()
        except ImportError:
            pass
        
        # Set seed again and generate new numbers
        set_deterministic_seed(42)
        np_random_new = np.random.random(10)
        
        # Should be identical
        np.testing.assert_array_equal(np_random, np_random_new)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        self.assertIsInstance(device, str)
        self.assertIn(device, ["cuda", "mps", "cpu"])
    
    def test_deidentify_data(self):
        """Test data de-identification."""
        # Create test data with potential identifiers
        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "name": ["John Doe", "Jane Smith", "Bob Johnson"],
            "claim_amount": [1000, 2000, 1500],
            "fraud_label": [0, 1, 0]
        })
        
        df_deid = deidentify_data(df, id_columns=["patient_id"])
        
        # Check that ID columns are removed
        self.assertNotIn("patient_id", df_deid.columns)
        self.assertIn("claim_amount", df_deid.columns)
        self.assertIn("fraud_label", df_deid.columns)


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            "claim_amount": [1000, 2000, 1500],
            "num_services": [3, 5, 4],
            "provider_rating": [4.5, 3.2, 4.8],
            "days_in_hospital": [2, 4, 3]
        })
    
    def test_interaction_features(self):
        """Test interaction feature creation."""
        df_eng = FeatureEngineer.create_interaction_features(self.df)
        
        self.assertIn("amount_per_service", df_eng.columns)
        self.assertIn("rating_per_day", df_eng.columns)
        self.assertGreater(len(df_eng.columns), len(self.df.columns))
    
    def test_risk_scores(self):
        """Test risk score creation."""
        df_risk = FeatureEngineer.create_risk_scores(self.df)
        
        self.assertIn("provider_risk", df_risk.columns)
        self.assertIn("amount_risk", df_risk.columns)
        self.assertGreater(len(df_risk.columns), len(self.df.columns))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestFraudDataset,
        TestModels,
        TestMetrics,
        TestUtils,
        TestFeatureEngineer
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
