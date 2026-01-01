"""
Streamlit demo for healthcare fraud detection.

This interactive demo allows users to explore fraud detection models,
upload data, make predictions, and visualize results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import FraudDataset, FeatureEngineer
from models import create_model, BaseFraudDetector
from metrics import FraudDetectionMetrics
from utils import set_deterministic_seed, suppress_warnings

# Configure page
st.set_page_config(
    page_title="Healthcare Fraud Detection AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
suppress_warnings()

# Set random seed
set_deterministic_seed(42)


def main():
    """Main Streamlit application."""
    
    # Header with disclaimer
    st.title("🏥 Healthcare Fraud Detection AI Research Demo")
    
    # Important disclaimer
    st.error("""
    **⚠️ IMPORTANT DISCLAIMER**
    
    This is a **RESEARCH DEMONSTRATION ONLY** - NOT FOR CLINICAL USE
    
    - This software is for research and educational purposes only
    - It is NOT intended for clinical diagnosis or medical decision-making
    - Results should NOT be used to make healthcare decisions
    - All healthcare decisions must be made by qualified medical professionals
    - This is NOT a validated medical device
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["xgboost", "lightgbm", "catboost", "isolation_forest", "tabnet"],
        help="Choose the fraud detection model to use"
    )
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload CSV File"],
        help="Choose whether to generate synthetic data or upload your own"
    )
    
    # Main content area
    if data_source == "Generate Synthetic Data":
        demo_synthetic_data(model_type)
    else:
        demo_upload_data(model_type)


def demo_synthetic_data(model_type: str):
    """Demo with synthetic data generation."""
    
    st.header("Synthetic Data Demo")
    
    # Parameters for synthetic data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.slider("Number of Samples", 1000, 10000, 5000)
        fraud_rate = st.slider("Fraud Rate (%)", 5, 20, 10) / 100
    
    with col2:
        random_seed = st.number_input("Random Seed", value=42, min_value=0)
        deidentify = st.checkbox("De-identify Data", value=True)
    
    with col3:
        if st.button("Generate New Dataset", type="primary"):
            st.session_state.generate_new_data = True
    
    # Generate dataset
    if not hasattr(st.session_state, 'dataset') or st.session_state.get('generate_new_data', False):
        with st.spinner("Generating synthetic dataset..."):
            # Generate synthetic data
            n_fraud = int(n_samples * fraud_rate)
            n_normal = n_samples - n_fraud
            
            np.random.seed(random_seed)
            
            # Normal claims
            normal_claims = pd.DataFrame({
                "claim_amount": np.random.lognormal(6, 0.5, n_normal),
                "num_services": np.random.poisson(3, n_normal),
                "provider_rating": np.random.uniform(3.5, 5.0, n_normal),
                "days_in_hospital": np.random.poisson(2, n_normal),
                "patient_age": np.random.choice(["18-30", "31-50", "51-70", "70+"], n_normal, p=[0.2, 0.3, 0.3, 0.2]),
                "diagnosis_code": np.random.choice(["A", "B", "C", "D", "E"], n_normal, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                "procedure_code": np.random.choice(["P1", "P2", "P3", "P4"], n_normal, p=[0.4, 0.3, 0.2, 0.1]),
                "fraud_label": 0
            })
            
            # Fraudulent claims
            fraud_claims = pd.DataFrame({
                "claim_amount": np.random.lognormal(7.5, 0.8, n_fraud),
                "num_services": np.random.poisson(8, n_fraud),
                "provider_rating": np.random.uniform(1.0, 3.0, n_fraud),
                "days_in_hospital": np.random.poisson(6, n_fraud),
                "patient_age": np.random.choice(["18-30", "31-50", "51-70", "70+"], n_fraud, p=[0.1, 0.2, 0.4, 0.3]),
                "diagnosis_code": np.random.choice(["A", "B", "C", "D", "E"], n_fraud, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
                "procedure_code": np.random.choice(["P1", "P2", "P3", "P4"], n_fraud, p=[0.1, 0.2, 0.3, 0.4]),
                "fraud_label": 1
            })
            
            # Combine and shuffle
            df = pd.concat([normal_claims, fraud_claims], ignore_index=True)
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            # Add engineered features
            df["claim_to_services_ratio"] = df["claim_amount"] / (df["num_services"] + 1)
            df["provider_efficiency"] = df["provider_rating"] / (df["days_in_hospital"] + 1)
            df["age_risk_score"] = df["patient_age"].map({"18-30": 1, "31-50": 2, "51-70": 3, "70+": 4})
            
            if deidentify:
                df = df.drop(columns=["patient_age"], errors="ignore")  # Remove potentially identifying info
            
            st.session_state.dataset = df
            st.session_state.generate_new_data = False
    
    # Display dataset info
    dataset = st.session_state.dataset
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(dataset))
    with col2:
        st.metric("Fraud Cases", dataset["fraud_label"].sum())
    with col3:
        st.metric("Fraud Rate", f"{dataset['fraud_label'].mean():.1%}")
    with col4:
        st.metric("Features", len(dataset.columns) - 1)
    
    # Data visualization
    st.subheader("Data Overview")
    
    # Feature distributions
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "fraud_label"]
    
    if len(numeric_cols) > 0:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=numeric_cols[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(numeric_cols[:4]):
            row = i // 2 + 1
            col_idx = i % 2 + 1
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=dataset[col],
                    name=col,
                    opacity=0.7,
                    nbinsx=30
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Train and evaluate model
    if st.button("Train Model", type="primary"):
        train_and_evaluate_model(dataset, model_type)


def demo_upload_data(model_type: str):
    """Demo with uploaded data."""
    
    st.header("Upload Data Demo")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV file with healthcare claims data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Display data info
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Check for required columns
            required_cols = ["fraud_label"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV has a 'fraud_label' column (0 for normal, 1 for fraud)")
            else:
                # Data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", len(df))
                with col2:
                    st.metric("Fraud Cases", df["fraud_label"].sum())
                with col3:
                    st.metric("Fraud Rate", f"{df['fraud_label'].mean():.1%}")
                with col4:
                    st.metric("Features", len(df.columns) - 1)
                
                # Train model
                if st.button("Train Model", type="primary"):
                    train_and_evaluate_model(df, model_type)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def train_and_evaluate_model(dataset: pd.DataFrame, model_type: str):
    """Train and evaluate the selected model."""
    
    with st.spinner(f"Training {model_type} model..."):
        try:
            # Prepare data
            fraud_dataset = FraudDataset(data=dataset, deidentify=True)
            X_train, X_val, X_test, y_train, y_val, y_test = fraud_dataset.get_train_test_split(
                test_size=0.2, val_size=0.1, random_state=42, stratify=True
            )
            
            # Create and train model
            model = create_model(model_type)
            
            if model_type == "tabnet":
                model = create_model(model_type, input_dim=X_train.shape[1])
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=50)
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics_calculator = FraudDetectionMetrics(threshold=0.5)
            metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred, y_proba)
            
            # Store results in session state
            st.session_state.model_results = {
                "model": model,
                "metrics": metrics,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "X_test": X_test
            }
            
            st.success("Model training completed!")
            
            # Display results
            display_model_results(metrics, y_test, y_pred, y_proba)
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")


def display_model_results(metrics: dict, y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    """Display model evaluation results."""
    
    st.header("Model Evaluation Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    
    with col2:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
    
    with col3:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
    
    with col4:
        st.metric("Specificity", f"{metrics.get('specificity', 0):.3f}")
        st.metric("Brier Score", f"{metrics.get('brier_score', 0):.3f}")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Precision-Recall", "Calibration", "Confusion Matrix"])
    
    with tab1:
        # ROC Curve
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        auc_score = roc_auc_score(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_score:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        pr_auc = average_precision_score(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR Curve (AUC = {pr_auc:.3f})'))
        
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Calibration Curve
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba, n_bins=10
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, mode='markers+lines', name='Model'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfectly Calibrated', line=dict(dash='dash')))
        
        fig.update_layout(
            title="Calibration Curve",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            width=600,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ["Value"]
    metrics_df = metrics_df.round(4)
    
    st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    main()
