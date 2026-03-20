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
from textwrap import dedent

# Third-party library imports
import dotenv
dotenv.load_dotenv()  # has to be executed before kaggle

import kaggle

from src.WasteClassifier.entity.config_entity import DataIngestionConfig
from src.WasteClassifier.utils.common import get_size

logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles the downloading of datasets for the Waste Classifier project,
    including validations, logging, and data size calculations.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialization with configuration provided.

        :param config: Configuration object for data ingestion.
        """
        self.config = config
    
    def _is_dataset_downloaded(self) -> bool:
        """
        Check if the dataset has already been downloaded.

        Returns:
            True if dataset exists and is non-empty, False otherwise.
        """
        if os.path.exists(self.config.local_data_file):
            logger.info("Dataset is already downloaded.")
            return True
        else:
            logger.info("Dataset is not downloaded. Proceeding to download.")
            return False


    def _download_file(self) -> None:
        """
        Handles dataset download from Kaggle and ensures it is available locally.
        
        Returns:
            None
            
        Raises:
            ValueError: If source_url is missing in the configuration.
            EnvironmentError: If Kaggle credentials are not found.
            RuntimeError: If the download fails.
        """
        logger.info("Starting dataset download process...")

        # Check if dataset is already downloaded
        if self._is_dataset_downloaded():
            dataset_size = get_size(Path(self.config.local_data_file))
            logger.info(f"Dataset already downloaded. Size: {dataset_size} kB")
            return

        # Attempt to download the dataset
        try:
            logger.info(f"Downloading dataset from Kaggle: {self.config.source_url}")
            
            # Try to authenticate with existing credentials
            try:
                kaggle.api.authenticate()
            except Exception as auth_error:
                logger.warning(f"Authentication failed: {auth_error}")
                
                # Try with access token from file
                access_token_path = Path.home() / ".kaggle" / "access_token"
                if access_token_path.exists():
                    with open(access_token_path, 'r') as f:
                        token = f.read().strip()
                    
                    # Set token as environment variable for kaggle API
                    os.environ['KAGGLE_API_TOKEN'] = token
                    logger.info("Using access token from file for authentication")
                    
                    # Try authentication again
                    kaggle.api.authenticate()
                else:
                    raise auth_error
            
            kaggle.api.dataset_download_files(
                dataset=self.config.source_url,
                path=self.config.root_dir,
                unzip=False
            )
            
            # Find the downloaded zip file and rename it
            zip_files = list(Path(self.config.root_dir).glob("*.zip"))
            if zip_files:
                os.rename(zip_files[0], self.config.local_data_file)
                logger.info(f"Dataset '{self.config.source_url}' has been successfully downloaded to '{self.config.local_data_file}'.")
            else:
                raise FileNotFoundError("No zip file found after download")
                
        except Exception as e:
            logger.error(f"Error occurred while downloading the dataset: {e}")
            raise

        # Verify download
        if self._is_dataset_downloaded():
            dataset_size = get_size(Path(self.config.local_data_file))
            logger.info(f"Dataset download complete. Size: {dataset_size} kB.")
        else:
            error_message = f"Dataset download failed. The file '{self.config.local_data_file}' was not created."
            logger.error(error_message)
            raise RuntimeError(error_message)
    
    def _extract_zip_file(self) -> None:
        """
        Extract zip file to specified directory.
        
        Returns:
            None
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
        self._download_file()
        self._extract_zip_file()
        logger.info("Data ingestion completed")