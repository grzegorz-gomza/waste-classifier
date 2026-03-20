"""
Prediction script for waste classification models.

This script allows users to:
1. Predict on a specific image path
2. Predict on a random image from the dataset
3. Display the image with true and predicted labels
4. Compare predictions from all trained models

Author: Grzegorz Gomza
Date: February 2026
"""

import logging
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.WasteClassifier.components.dataset import WasteDataset

logger = logging.getLogger(__name__)

class WasteClassifierPredictor:
    def __init__(self, models_dir="artifacts/training_dl"):
        """
        Initialize the predictor.
        
        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_names = []
        self.transform = None
        
        logger.info(f"Using device: {self.device}")
        
        # Load models and class names
        self._load_models()
        self._setup_transform()
    
    def _load_models(self):
        """Load all trained models."""
        model_names = ['resnet50', 'mobilenet_v2', 'efficientnet_b0']
        
        for model_name in model_names:
            model_path = self.models_dir / f"{model_name}_best_model.pth"
            if model_path.exists():
                try:
                    model = torch.load(model_path, weights_only=False)
                    model = model.to(self.device)
                    model.eval()
                    self.models[model_name] = model
                    logger.info(f"✅ Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name} model: {e}")
            else:
                logger.warning(f"⚠️ {model_name} model not found at {model_path}")
        
        if not self.models:
            raise RuntimeError("No models found! Please train models first.")
        
        # Load class names from dataset
        self._load_class_names()
    
    def _load_class_names(self):
        """Load class names from the dataset."""
        try:
            # Try to load from the data directory
            data_dir = Path("artifacts/data_ingestion") / "images" / "images"
            if data_dir.exists():
                classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
                self.class_names = classes
                logger.info(f"✅ Loaded {len(self.class_names)} class names")
            else:
                # Fallback to generic class names
                num_classes = 30  # Default from params.yaml
                self.class_names = [f"Class_{i}" for i in range(num_classes)]
                logger.warning(f"⚠️ Using generic class names (data directory not found)")
        except Exception as e:
            logger.error(f"❌ Error loading class names: {e}")
            self.class_names = [f"Class_{i}" for i in range(30)]
    
    def _setup_transform(self):
        """Setup image transforms for prediction."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def predict_image(self, image_path):
        """
        Predict on a single image using all models.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Predictions from all models
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            predictions = {}
            
            with torch.no_grad():
                for model_name, model in self.models.items():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    predictions[model_name] = {
                        'predicted_class_idx': predicted.item(),
                        'predicted_class': self.class_names[predicted.item()],
                        'confidence': confidence.item(),
                        'all_probabilities': probabilities.cpu().numpy().flatten()
                    }
            
            return predictions, image
            
        except Exception as e:
            logger.error(f"❌ Error predicting image {image_path}: {e}")
            return None, None
    
    def get_random_test_image(self):
        """
        Get a random image from the test dataset.
        
        Returns:
            tuple: (image_path, true_class_name) or (None, None) if no dataset found
        """
        try:
            # Try to load test dataset
            test_dataset = WasteDataset(
                root_dir="artifacts/data_ingestion",
                transform=None,  # No transform for getting original image
                split='test',
                test_split=0.2
            )
            
            if len(test_dataset) == 0:
                logger.warning("⚠️ No test images found")
                return None, None
            
            # Get random sample
            idx = random.randint(0, len(test_dataset) - 1)
            img_path, label_idx = test_dataset.samples[idx]
            true_class = test_dataset.get_class_names()[label_idx]
            
            return img_path, true_class
            
        except Exception as e:
            logger.error(f"❌ Error getting random test image: {e}")
            return None, None
    
    def display_predictions(self, image, predictions, image_path=None, true_class=None):
        """
        Display image with predictions from all models.
        
        Args:
            image: PIL Image
            predictions: dict of predictions from all models
            image_path: path to the image (optional)
            true_class: true class name (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Waste Classification Predictions', fontsize=16, fontweight='bold')
        
        # Main image display
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        if true_class:
            axes[0, 0].set_xlabel(f'True Class: {true_class}', fontsize=12, color='green')
        
        # Predictions summary
        axes[0, 1].axis('off')
        summary_text = "Model Predictions:\n\n"
        
        colors = ['blue', 'orange', 'green']
        for i, (model_name, pred) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            marker = "✓" if pred['predicted_class'] == true_class else "✗"
            summary_text += f"{marker} {model_name.upper()}: {pred['predicted_class']}\n"
            summary_text += f"   Confidence: {pred['confidence']:.3f}\n\n"
        
        axes[0, 1].text(0.1, 0.9, summary_text, transform=axes[0, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace')
        axes[0, 1].set_title('Prediction Summary', fontweight='bold')
        
        # Confidence comparison
        model_names = list(predictions.keys())
        confidences = [predictions[m]['confidence'] for m in model_names]
        
        bars = axes[1, 0].bar(model_names, confidences, color=colors[:len(model_names)])
        axes[1, 0].set_title('Confidence Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Top-5 probabilities for best model
        best_model = max(predictions.keys(), key=lambda x: predictions[x]['confidence'])
        best_probs = predictions[best_model]['all_probabilities']
        
        # Get top 5 predictions
        top5_idx = np.argsort(best_probs)[-5:][::-1]
        top5_classes = [self.class_names[i] for i in top5_idx]
        top5_probs = best_probs[top5_idx]
        
        y_pos = np.arange(len(top5_classes))
        axes[1, 1].barh(y_pos, top5_probs, color='skyblue')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(top5_classes)
        axes[1, 1].set_xlabel('Probability')
        axes[1, 1].set_title(f'Top-5 Predictions ({best_model.upper()})', fontweight='bold')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("artifacts/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = plt.gca().figure.number
        output_path = output_dir / f"prediction_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"📊 Prediction visualization saved to {output_path}")
        plt.show()
        
        return output_path
    
    def run_interactive(self):
        """Run interactive prediction session."""
        print("\n" + "="*60)
        print("🤖 WASTE CLASSIFICATION PREDICTION TOOL")
        print("="*60)
        print(f"📋 Available models: {', '.join(self.models.keys())}")
        print(f"🏷️  Number of classes: {len(self.class_names)}")
        
        while True:
            print("\n" + "-"*40)
            print("Choose an option:")
            print("1. Predict on a specific image path")
            print("2. Predict on a random test image")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                image_path = input("Enter image path: ").strip()
                if not os.path.exists(image_path):
                    print("❌ Image file not found!")
                    continue
                
                predictions, image = self.predict_image(image_path)
                if predictions:
                    self.display_predictions(image, predictions, image_path)
                    
            elif choice == '2':
                print("🎲 Getting random test image...")
                img_path, true_class = self.get_random_test_image()
                
                if img_path is None:
                    print("❌ No test images available!")
                    continue
                
                print(f"📷 Selected: {img_path}")
                print(f"🏷️  True class: {true_class}")
                
                predictions, image = self.predict_image(img_path)
                if predictions:
                    self.display_predictions(image, predictions, img_path, true_class)
                    
            elif choice == '3':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice! Please enter 1-3.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Waste Classification Prediction Tool")
    parser.add_argument("--image", type=str, help="Path to image file for prediction")
    parser.add_argument("--random", action="store_true", help="Use random test image")
    parser.add_argument("--models-dir", type=str, default="artifacts/training_dl", 
                       help="Directory containing trained models")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = WasteClassifierPredictor(args.models_dir)
        
        if args.image:
            # Predict on specific image
            if not os.path.exists(args.image):
                print(f"Image file not found: {args.image}")
                return
            
            predictions, image = predictor.predict_image(args.image)
            if predictions:
                predictor.display_predictions(image, predictions, args.image)
                
        elif args.random:
            # Predict on random image
            img_path, true_class = predictor.get_random_test_image()
            if img_path:
                print(f"Random image: {img_path}")
                print(f"True class: {true_class}")
                
                predictions, image = predictor.predict_image(img_path)
                if predictions:
                    predictor.display_predictions(image, predictions, img_path, true_class)
            else:
                print("No test images available!")
                
        else:
            # Run interactive mode
            predictor.run_interactive()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
