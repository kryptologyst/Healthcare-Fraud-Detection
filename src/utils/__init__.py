"""
Core utilities for healthcare fraud detection project.

This module provides common utilities including seeding, device management,
logging, and data validation functions.
"""

import random
import logging
import warnings
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame has required columns and no missing values.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If validation fails
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        raise ValueError(f"Missing values found in columns: {missing_values[missing_values > 0].to_dict()}")


def deidentify_data(df: pd.DataFrame, id_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove or hash potentially identifying information.
    
    Args:
        df: DataFrame to de-identify
        id_columns: Columns to remove or hash
        
    Returns:
        De-identified DataFrame
    """
    df_deid = df.copy()
    
    if id_columns is None:
        # Common ID column patterns
        id_patterns = ["id", "patient_id", "provider_id", "claim_id", "ssn", "mrn"]
        id_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in id_patterns)]
    
    # Remove ID columns
    df_deid = df_deid.drop(columns=id_columns, errors="ignore")
    
    # Hash remaining potentially identifying columns
    hash_columns = ["name", "address", "phone", "email"]
    for col in df_deid.columns:
        if any(pattern in col.lower() for pattern in hash_columns):
            df_deid[col] = df_deid[col].astype(str).apply(lambda x: hash(x) % 1000000)
    
    return df_deid


def suppress_warnings() -> None:
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """Create standard directory structure for the project.
    
    Args:
        base_path: Base project path
    """
    base_path = Path(base_path)
    directories = [
        "data",
        "models", 
        "configs",
        "scripts",
        "demo",
        "tests",
        "assets",
        "notebooks",
        "logs"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(exist_ok=True)
        # Create .gitkeep files
        (base_path / directory / ".gitkeep").touch()


class ConfigManager:
    """Configuration manager for handling model and training configurations."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.base_config = self.config.copy()
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration.
        
        Args:
            model_type: Type of model
            
        Returns:
            Model configuration dictionary
        """
        if model_type not in self.config.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return OmegaConf.to_container(self.config.models[model_type], resolve=True)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration.
        
        Returns:
            Training configuration dictionary
        """
        return OmegaConf.to_container(self.config.training, resolve=True)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration.
        
        Returns:
            Data configuration dictionary
        """
        return OmegaConf.to_container(self.config.data, resolve=True)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config = OmegaConf.merge(self.config, OmegaConf.create(updates))
    
    def reset_config(self) -> None:
        """Reset configuration to base values."""
        self.config = self.base_config.copy()
