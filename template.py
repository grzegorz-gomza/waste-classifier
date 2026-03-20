"""
Template generator for project structure.

Author: Grzegorz Gomza
Date: February 2026
"""

import os
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Project name
project_name = "WasteClassifier"

# List of files to create
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",

    # Components
    # Shared Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/share/__init__.py",
    f"src/{project_name}/components/share/data_ingestion.py", # Shared data ingestion component
    f"src/{project_name}/components/share/dataset.py", # Shared dataset class

    # Deep Learning Components
    f"src/{project_name}/components/deep_learning/__init__.py",
    f"src/{project_name}/components/deep_learning/prepare_base_model.py", # Preparing DL base model for transfer learning 
    f"src/{project_name}/components/deep_learning/train.py", # Training DL model
    f"src/{project_name}/components/deep_learning/evaluate.py", # Evaluating DL model

    # Machine Learning Components
    f"src/{project_name}/components/machine_learning/__init__.py",
    f"src/{project_name}/components/machine_learning/feature_engineering.py", # Feature extraction for ML models
    f"src/{project_name}/components/machine_learning/train.py", # Training ML models
    f"src/{project_name}/components/machine_learning/evaluate.py", # Evaluating ML models

    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py", # Common-used utils

    # Configuration
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py", # Main configuration system
    
    # Pipelines
    f"src/{project_name}/pipeline/__init__.py",
    
    # Deep Learning Pipelines
    f"src/{project_name}/pipeline/deep_learning/__init__.py",
    f"src/{project_name}/pipeline/deep_learning/stage_01_data_ingestion.py", # DL data ingestion
    f"src/{project_name}/pipeline/deep_learning/stage_02_prepare_models.py", # Prepare DL base model
    f"src/{project_name}/pipeline/deep_learning/stage_03_train.py", # Train DL model
    f"src/{project_name}/pipeline/deep_learning/stage_04_evaluate.py", # Evaluate DL model
    
    # Machine Learning Pipelines
    f"src/{project_name}/pipeline/machine_learning/__init__.py",
    f"src/{project_name}/pipeline/machine_learning/stage_01_data_ingestion.py", # ML data ingestion
    f"src/{project_name}/pipeline/machine_learning/stage_02_feature_engineering.py", # Feature engineering
    f"src/{project_name}/pipeline/machine_learning/stage_03_train.py", # Train ML models
    f"src/{project_name}/pipeline/machine_learning/stage_04_evaluate.py", # Evaluate ML models
    
    # Configuration Entity
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    
    # Project constants
    f"src/{project_name}/constants/__init__.py",

    # Project configuration files
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",
    
    # Entry point for training 
    "main.py",
    
    # Docker
    "Dockerfile",
    
    # App initial files for JavaScript
    "frontend/package.json",
    "frontend/src/App.js",
    "frontend/src/index.js",
    "backend/server.js",

    # Python tests
    "tests/__init__.py",
    
    # Log messages file
    "log_msgs/created_items.json",
]

def create_project_structure():
    # Track created files and directories
    created_items = {"directories": {
        "created": [],
        "skipped": []
        }, "files": {
                "created": [],
                "skipped": []
        }}
    
    # Create project files and directories
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath) # Split the path into directory and filename
    
        # Create directory if it's not empty
        if filedir != "":
            if os.path.exists(filedir):
                logging.info(f"Directory {filedir} already exists")
                created_items["directories"]["skipped"].append(filedir)
            else:
                os.makedirs(filedir, exist_ok=True)
                logging.info(f"Creating directory: {filedir}")
                created_items["directories"]["created"].append(filedir)
        
        # Create file if it doesn't exist
        if os.path.exists(filepath) or os.path.getsize(filepath) > 0:
            logging.info(f"File {filename} already exists")
            created_items["files"]["skipped"].append(str(filepath))
        else:
            with open(filepath, "w"):
                pass
            logging.info(f"Creating empty file: {filepath}")
            created_items["files"]["created"].append(str(filepath))
    
    # Save created items to log file
    with open("log_msgs/created_items.json", "w") as f:
        json.dump(created_items, f, indent=4)
    logging.info("Created items saved to log_msgs/created_items.json")

if __name__ == "__main__":
    try:
        logging.info("Project structure creation started...")
        create_project_structure()
        logging.info("Project structure created successfully!")
    except Exception as e:
        logging.error(f"Error creating project structure: {e}")
        raise
