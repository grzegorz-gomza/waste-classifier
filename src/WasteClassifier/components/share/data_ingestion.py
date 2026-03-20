"""
Data ingestion component for downloading and extracting Kaggle dataset.

Author: Grzegorz Gomza
Date: February 2026
References:
- Kaggle API: https://github.com/Kaggle/kaggle-api
"""

import os
import zipfile
import logging
from pathlib import Path
from WasteClassifier.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion component.
        
        Args:
            config (DataIngestionConfig): Configuration for data ingestion
        """
        self.config = config
    
    def download_file(self):
        """
        Download dataset from Kaggle using Kaggle API.
        Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info("Downloading data from Kaggle...")
            
            # Download using Kaggle API
            os.system(f"kaggle datasets download -d {self.config.source_url} -p {self.config.root_dir}")
            
            # Find the downloaded zip file
            zip_files = list(Path(self.config.root_dir).glob("*.zip"))
            if zip_files:
                os.rename(zip_files[0], self.config.local_data_file)
                logger.info(f"Downloaded data to {self.config.local_data_file}")
            else:
                raise FileNotFoundError("No zip file found after download")
        else:
            logger.info(f"File already exists: {self.config.local_data_file}")
    
    def extract_zip_file(self):
        """
        Extract zip file to specified directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        # Check if zip file exists
        if not os.path.exists(self.config.local_data_file):
            logger.error(f"Zip file not found: {self.config.local_data_file}")
            raise FileNotFoundError(f"Zip file not found: {self.config.local_data_file}")
        
        # Check if data is already extracted (look for image files)
        image_files = list(Path(unzip_path).rglob("*.jpg")) + list(Path(unzip_path).rglob("*.png"))
        if image_files:
            logger.info(f"Data already extracted (found {len(image_files)} image files)")
            return
        
        # Extract if no images found
        logger.info(f"Extracting {self.config.local_data_file} to {unzip_path}")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info("Extraction completed")

    def main(self):
        """Main method to run data ingestion pipeline."""
        self.download_file()
        self.extract_zip_file()
        logger.info("Data ingestion completed")