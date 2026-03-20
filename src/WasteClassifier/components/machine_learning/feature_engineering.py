"""
Feature engineering techniques for waste classification.

This module provides various feature extraction methods for image data
to enhance XGBoost model performance.

Author: Grzegorz Gomza
Date: March 2026
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import img_as_float
from skimage.color import rgb2gray
import torch
from torchvision import transforms
from PIL import Image
import pickle
import os
from pathlib import Path
import hashlib
import dask
import dask.array as da

logger = logging.getLogger(__name__)


class FeatureEngineeringMaster:
    """
    Master class for feature engineering techniques applied to waste classification.
    
    This class implements various feature extraction methods including:
    - Local maxima detection
    - Statistical features
    - Texture features
    - Shape features
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature engineering master.
        
        Args:
            config: Configuration dictionary for feature engineering parameters (ignored)
        """
        # Set fixed parameters
        self.min_distance = 10
        self.max_filter_size = 10
        
        # Setup caching
        self.cache_dir = Path("artifacts/feature_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Feature cache directory: {self.cache_dir}")
        logger.info(f"Fixed parameters: min_distance={self.min_distance}, max_filter_size={self.max_filter_size}")
        
    def _get_image_hash(self, image: np.ndarray) -> str:
        """Generate hash for image caching."""
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def _get_cache_path(self, image_hash: str) -> Path:
        """Get cache file path for image hash."""
        return self.cache_dir / f"{image_hash}.pkl"
    
    def _is_cached(self, image_hash: str) -> bool:
        """Check if features are cached for this image."""
        cache_path = self._get_cache_path(image_hash)
        return cache_path.exists()
    
    def _load_cached_features(self, image_hash: str) -> Optional[np.ndarray]:
        """Load cached features if available."""
        cache_path = self._get_cache_path(image_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                logger.debug(f"Loaded cached features: {cache_path}")
                return features
            except Exception as e:
                logger.warning(f"Failed to load cached features {cache_path}: {e}")
        return None
    
    def _save_cached_features(self, image_hash: str, features: np.ndarray) -> None:
        """Save features to cache."""
        cache_path = self._get_cache_path(image_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            logger.debug(f"Saved cached features: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cached features {cache_path}: {e}")
        
    def extract_local_maxima_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract local maxima features using peak_local_max directly with caching.
        Creates a fixed 500-feature vector where each position represents peak coordinates.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            
        Returns:
            Fixed-size numpy array of 500 features representing peak coordinates
        """
        # Check cache first
        image_hash = self._get_image_hash(image)
        
        # Try to load from cache
        cached_features = self._load_cached_features(image_hash)
        if cached_features is not None:
            return cached_features
        
        # If not cached, extract features
        logger.debug(f"Extracting features for image hash: {image_hash}")
        
        # Convert to grayscale using scikit-image rgb2gray
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB
                image = rgb2gray(image)
            else:
                image = image[:, :, 0]  # Take first channel
        else:
            image = image
        
        # Ensure image is in float format
        image = img_as_float(image)
        
        # Use peak_local_max directly to get peak coordinates
        # Limit number of peaks for speed - we only need first 250 anyway
        coordinates = peak_local_max(image, min_distance=self.min_distance, num_peaks=500)
        
        # Create fixed-size vector of 500 features
        # Each feature represents a potential peak location (x, y coordinates)
        feature_vector = np.zeros(500)
        
        # Fill vector with peak coordinates
        # Each peak occupies 2 consecutive positions: [x1, y1, x2, y2, ...]
        num_peaks_to_use = min(len(coordinates), 250)  # Max 250 peaks = 500 features
        
        for i in range(num_peaks_to_use):
            if i < len(coordinates):
                y_coord, x_coord = coordinates[i]
                feature_vector[2*i] = x_coord      # x coordinate
                feature_vector[2*i + 1] = y_coord  # y coordinate
            else:
                # If no more peaks, fill with zeros
                feature_vector[2*i] = 0.0
                feature_vector[2*i + 1] = 0.0
        
        # Save to cache
        self._save_cached_features(image_hash, feature_vector)
        
        return feature_vector
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all features from an image and return as a feature vector.
        Now only uses local maxima coordinates (500 features).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Feature vector as numpy array (500 features representing peak coordinates)
        """
        # Extract local maxima features (500 features)
        peak_features = self.extract_local_maxima_features(image)
        
        return peak_features
    
    def visualize_local_maxima(self, image: np.ndarray, save_path: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize local maxima detection on an image following the example approach.
        
        Args:
            image: Input image as numpy array
            save_path: Optional path to save the visualization
            
        Returns:
            Tuple of (figure, coordinates)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray_image = image[:, :, 0]
        else:
            gray_image = image
        
        # Ensure image is in float format
        gray_image = img_as_float(gray_image)
        
        # Apply maximum filter as shown in the example
        # image_max is the dilation of im with a 20*20 structuring element
        image_max = ndi.maximum_filter(gray_image, size=self.max_filter_size, mode='constant')
        
        # Find local maxima coordinates
        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(gray_image, min_distance=self.min_distance)
        
        # Create visualization matching the example exactly
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        ax = axes.ravel()
        
        # Original image
        ax[0].imshow(gray_image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')
        
        # Maximum filter
        ax[1].imshow(image_max, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Maximum filter')
        
        # Peak local max
        ax[2].imshow(gray_image, cmap=plt.cm.gray)
        ax[2].autoscale(False)
        ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        ax[2].axis('off')
        ax[2].set_title(f'Peak local max ({len(coordinates)} peaks)')
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Local maxima visualization saved to {save_path}")
        
        return fig, coordinates
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features that will be extracted.
        Now returns 500 feature names representing peak coordinates.
        
        Returns:
            List of feature names
        """
        # Generate 500 feature names: peak_1_x, peak_1_y, peak_2_x, peak_2_y, ...
        feature_names = []
        for i in range(1, 251):  # 250 peaks = 500 features
            feature_names.append(f"peak_{i}_x")
            feature_names.append(f"peak_{i}_y")
        
        return feature_names


def create_sample_visualization(dataset_path: str, save_path: str = "local_maxima_demo.png"):
    """
    Create a sample visualization of local maxima detection on waste dataset image.
    
    Args:
        dataset_path: Path to the waste dataset
        save_path: Path to save the visualization
    """
    try:
        from WasteClassifier.components.share.dataset import WasteDataset
        
        # Load a sample image
        dataset = WasteDataset(
            root_dir=dataset_path,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
            ]),
        )
        
        # Get first image
        sample_image, _ = dataset[0]
        
        # Convert to numpy array
        if isinstance(sample_image, torch.Tensor):
            sample_image = sample_image.numpy()
        
        # Transpose from (C, H, W) to (H, W, C) if needed
        if len(sample_image.shape) == 3 and sample_image.shape[0] == 3:
            sample_image = np.transpose(sample_image, (1, 2, 0))
        
        # Create feature engineering instance
        fe_master = FeatureEngineeringMaster()
        
        # Create visualization
        fig, coordinates = fe_master.visualize_local_maxima(sample_image, save_path)
        
        # Extract and print features
        features = fe_master.extract_local_maxima_features(sample_image)
        print(f"Found {len(coordinates)} local maxima")
        print(f"Feature vector shape: {fe_master.extract_all_features(sample_image).shape}")
        print(f"Feature names: {fe_master.get_feature_names()}")
        
        plt.show()
        
        return fig, coordinates, features
        
    except Exception as e:
        logger.error(f"Error creating sample visualization: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample visualization with a test image
    from skimage import data
    
    # Use coins image as example
    coins = data.coins()
    fe_master = FeatureEngineeringMaster()
    
    # Visualize local maxima
    fig, coords = fe_master.visualize_local_maxima(coins, "local_maxima_coins_demo.png")
    
    # Extract features
    features = fe_master.extract_all_features(coins)
    print(f"Extracted {len(features)} features from coins image")
    print(f"Feature names: {fe_master.get_feature_names()}")
    
    plt.show()
