# Healthcare Fraud Detection AI Research Project

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH DEMONSTRATION PROJECT ONLY**

- **NOT FOR CLINICAL USE**: This project is for research and educational purposes only
- **NOT MEDICAL ADVICE**: Do not use for diagnostic, therapeutic, or clinical decision-making
- **NO WARRANTY**: Results are not guaranteed and should not be relied upon for any medical purpose
- **CLINICIAN SUPERVISION REQUIRED**: All healthcare decisions must be made by qualified medical professionals
- **RESEARCH ONLY**: This is a demonstration of AI techniques, not a validated medical device

## Overview

This project implements healthcare fraud detection using machine learning techniques on simulated insurance claims data. It demonstrates various approaches including unsupervised anomaly detection, supervised classification, and deep learning methods for identifying potentially fraudulent healthcare claims.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Quick Demo**:
   ```bash
   python quickstart.py
   ```

3. **Launch Interactive Demo**:
   ```bash
   streamlit run demo/app.py
   ```

4. **Train Custom Model**:
   ```bash
   python scripts/train.py --model-type xgboost --config configs/xgboost.yaml
   ```

## Features

- **Multiple ML Approaches**: Isolation Forest, Gradient Boosting, Deep Tabular Networks
- **Comprehensive Evaluation**: Clinical metrics, calibration analysis, fairness assessment
- **Explainability**: SHAP explanations, feature importance analysis
- **Uncertainty Quantification**: Confidence intervals and calibration curves
- **Interactive Demo**: Streamlit web interface for exploration
- **Privacy Protection**: Built-in de-identification capabilities

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing utilities
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
├── configs/               # Configuration files
├── data/                  # Data directory
├── models/                # Saved models
├── scripts/               # Training/evaluation scripts
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── notebooks/             # Jupyter notebooks
├── quickstart.py          # Quick start script
└── requirements.txt       # Dependencies
```

## Models

1. **Isolation Forest**: Unsupervised anomaly detection
2. **XGBoost**: Gradient boosting classifier
3. **LightGBM**: Gradient boosting classifier
4. **CatBoost**: Gradient boosting classifier
5. **TabNet**: Deep tabular neural network

## Evaluation Metrics

- **Classification**: AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV
- **Calibration**: Brier Score, Expected Calibration Error
- **Fairness**: Performance across demographic groups
- **Explainability**: SHAP feature importance, LIME explanations

## Usage Examples

### Training a Model
```python
from src.train import train_model
from src.data import FraudDataset

# Load data
dataset = FraudDataset("data/synthetic_claims.csv")

# Train model
model = train_model(
    dataset=dataset,
    model_type="xgboost",
    config_path="configs/xgboost.yaml"
)
```

### Making Predictions
```python
from src.models import FraudDetector

# Load trained model
detector = FraudDetector.load("models/best_model.pkl")

# Predict fraud probability
claims = pd.read_csv("data/new_claims.csv")
predictions = detector.predict_proba(claims)
```

## Configuration

Models can be configured using YAML files in the `configs/` directory:

```yaml
model:
  type: "xgboost"
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42

training:
  batch_size: 32
  epochs: 100
  early_stopping: 10
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{healthcare_fraud_detection,
  title={Healthcare Fraud Detection AI Research Project},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Healthcare-Fraud-Detection}
}
```

## Acknowledgments

- Built for educational and research purposes
- Uses synthetic data to avoid privacy concerns
- Implements state-of-the-art fraud detection techniques
- Designed with explainability and fairness in mind
# Healthcare-Fraud-Detection
