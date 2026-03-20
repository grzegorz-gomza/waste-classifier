"""
Deep learning model evaluation component.

Author: Grzegorz Gomza
Date: February 2026
References:
- PyTorch evaluation: https://pytorch.org/tutorials/beginner/basics/evaluation_tutorial.html
- Metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os
from pathlib import Path

from WasteClassifier.entity.config_entity import EvaluationConfig
from WasteClassifier.components.share.dataset import WasteDataset

logger = logging.getLogger(__name__)

class EvaluateDLModel:
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation component.
        
        Args:
            config (EvaluationConfig): Evaluation configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create timestamped report directory
        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")
        self.report_dir = Path("artifacts") / "reports" / timestamp
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report directory: {self.report_dir}")
    
    def get_training_params(self):
        """
        Get training parameters from params.yaml.
        
        Returns:
            dict: Training parameters
        """
        try:
            params_path = Path("params.yaml")
            if params_path.exists():
                import yaml
                with open(params_path, 'r') as f:
                    params = yaml.safe_load(f)
                return params
            else:
                logger.warning("params.yaml not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading params.yaml: {e}")
            return {}
    
    def load_models(self):
        """
        Load all trained DL models for evaluation.
        
        Returns:
            dict: Dictionary of loaded models
        """
        logger.info("Loading DL models for evaluation...")
        
        models_to_load = ['resnet50', 'mobilenet_v2', 'efficientnet_b0']
        models = {}
        
        for model_name in models_to_load:
            model_path = self.config.dl_model_path.parent / f"{model_name}_best_model.pth"
            if model_path.exists():
                try:
                    model = torch.load(model_path, weights_only=False)
                    model = model.to(self.device)
                    model.eval()
                    models[model_name] = model
                    logger.info(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {e}")
            else:
                logger.warning(f"{model_name} model not found at {model_path}")
        
        return models
    
    def get_test_data_loader(self):
        """
        Create test data loader.
        
        Returns:
            DataLoader: Test data loader
        """
        # Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize(tuple(self.config.params_image_size[:2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Use test split for evaluation
        test_dataset = WasteDataset(
            root_dir=self.config.test_data_path,
            transform=transform,
            split='test',
            test_split=0.2  # This parameter is not used when split='test'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return test_loader, test_dataset
    
    def evaluate_models(self, models, test_loader):
        """
        Evaluate all models performance.
        
        Args:
            models: Dictionary of trained models
            test_loader: Test data loader
            
        Returns:
            dict: Evaluation metrics for all models
        """
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                    images = images.to(self.device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            report = classification_report(all_labels, all_preds, output_dict=True)
            cm = confusion_matrix(all_labels, all_preds)
            
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
            
            logger.info(f"{model_name} Model Accuracy: {accuracy:.4f}")
        
        return results
    
    def save_results(self, results, test_dataset):
        """
        Save comprehensive evaluation results with plots and parameters.
        
        Args:
            results: Evaluation metrics for all models
            test_dataset: Test dataset for class names
        """
        class_names = test_dataset.get_class_names()
        training_params = self.get_training_params()
        
        # Save combined metrics as JSON
        metrics_path = self.report_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save training parameters
        params_path = self.report_dir / 'training_parameters.json'
        with open(params_path, 'w') as f:
            json.dump(training_params, f, indent=2)
        
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        logger.info(f"Training parameters saved to {params_path}")
        
        # Create comprehensive comparison plots
        self._create_comparison_plots(results, class_names)
        
        # Create individual confusion matrices
        self._create_confusion_matrices(results, class_names)
        
        # Create performance summary
        self._create_performance_summary(results, training_params)
        
        logger.info(f"All reports saved to {self.report_dir}")
    
    def _create_comparison_plots(self, results, class_names):
        """Create comparison plots for all models."""
        
        # 1. Accuracy comparison bar chart
        accuracies = {model: metrics['accuracy'] for model, metrics in results.items()}
        
        plt.figure(figsize=(10, 6))
        models = list(accuracies.keys())
        acc_values = list(accuracies.values())
        
        bars = plt.bar(models, acc_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, acc_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision, Recall, F1 comparison
        metrics_comparison = {}
        for model_name, metrics in results.items():
            report = metrics['classification_report']
            metrics_comparison[model_name] = {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1-score': report['weighted avg']['f1-score']
            }
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Precision
        prec_values = [metrics_comparison[m]['precision'] for m in models]
        ax1.bar(models, prec_values, color='#ff7f0e')
        ax1.set_title('Weighted Precision Comparison', fontweight='bold')
        ax1.set_ylabel('Precision')
        ax1.set_ylim(0, 1)
        
        # Recall
        rec_values = [metrics_comparison[m]['recall'] for m in models]
        ax2.bar(models, rec_values, color='#2ca02c')
        ax2.set_title('Weighted Recall Comparison', fontweight='bold')
        ax2.set_ylabel('Recall')
        ax2.set_ylim(0, 1)
        
        # F1-Score
        f1_values = [metrics_comparison[m]['f1-score'] for m in models]
        ax3.bar(models, f1_values, color='#d62728')
        ax3.set_title('Weighted F1-Score Comparison', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.report_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-class performance heatmap
        per_class_metrics = {}
        for model_name, metrics in results.items():
            report = metrics['classification_report']
            per_class_metrics[model_name] = []
            for class_name in class_names:
                if class_name in report:
                    per_class_metrics[model_name].append(report[class_name]['f1-score'])
                else:
                    per_class_metrics[model_name].append(0.0)
        
        plt.figure(figsize=(15, 10))
        per_class_array = np.array(list(per_class_metrics.values()))
        
        sns.heatmap(per_class_array, 
                   xticklabels=class_names, 
                   yticklabels=list(per_class_metrics.keys()),
                   annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'F1-Score'})
        plt.title('Per-Class F1-Score Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Waste Categories')
        plt.ylabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.report_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrices(self, results, class_names):
        """Create individual confusion matrices for each model."""
        
        for model_name, metrics in results.items():
            cm = np.array(metrics['confusion_matrix'])
            
            # Create larger figure for better readability
            plt.figure(figsize=(20, 16))
            
            # Use raw counts instead of normalized for clarity
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'},
                       annot_kws={'size': 8})  # Smaller font for annotations
            
            plt.title(f'{model_name.upper()} Model - Confusion Matrix', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
            plt.ylabel('True Label', fontsize=14, labelpad=10)
            
            # Rotate labels for better fit
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout(pad=3.0)
            
            # Save with higher DPI for better quality
            plt.savefig(self.report_dir / f'confusion_matrix_{model_name}.png', 
                       dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"{model_name} confusion matrix saved")
    
    def _create_performance_summary(self, results, training_params):
        """Create a comprehensive performance summary."""
        
        summary_text = f"""
WASTE CLASSIFICATION MODEL EVALUATION REPORT
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRAINING PARAMETERS:
-------------------
Image Size: {training_params.get('IMAGE_SIZE', 'N/A')}
Batch Size: {training_params.get('BATCH_SIZE', 'N/A')}
Learning Rate: {training_params.get('LEARNING_RATE', 'N/A')}
Epochs: {training_params.get('EPOCHS', 'N/A')}
Optimizer: {training_params.get('OPTIMIZER', 'N/A')}
Data Augmentation: {training_params.get('AUGMENTATION', 'N/A')}
Pretrained: {training_params.get('PRETRAINED', 'N/A')}
Freeze Base: {training_params.get('FREEZE_BASE', 'N/A')}

MODEL PERFORMANCE:
-----------------
"""
        
        # Sort models by accuracy
        sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            report = metrics['classification_report']
            summary_text += f"""
{i}. {model_name.upper()}
   Accuracy: {metrics['accuracy']:.4f}
   Precision (weighted): {report['weighted avg']['precision']:.4f}
   Recall (weighted): {report['weighted avg']['recall']:.4f}
   F1-Score (weighted): {report['weighted avg']['f1-score']:.4f}
"""
        
        best_model = sorted_models[0][0]
        summary_text += f"""
BEST PERFORMING MODEL: {best_model.upper()}
Accuracy: {sorted_models[0][1]['accuracy']:.4f}

FILES GENERATED:
---------------
- accuracy_comparison.png: Model accuracy comparison
- metrics_comparison.png: Precision, Recall, F1 comparison
- per_class_performance.png: Per-class F1-score heatmap
- confusion_matrix_*.png: Individual confusion matrices
- evaluation_metrics.json: Detailed metrics for all models
- training_parameters.json: Training configuration
- performance_summary.txt: This summary file
"""
        
        # Save summary
        with open(self.report_dir / 'performance_summary.txt', 'w') as f:
            f.write(summary_text)
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        for model_name, metrics in sorted_models:
            logger.info(f"{model_name}: {metrics['accuracy']:.4f} accuracy")
        
        logger.info(f"\n🏆 Best performing model: {best_model} with {sorted_models[0][1]['accuracy']:.4f} accuracy")
        
        # Log detailed metrics for best model
        best_metrics = sorted_models[0][1]['classification_report']
        logger.info(f"\n{best_model.upper()} Detailed Performance:")
        logger.info(f"Precision: {best_metrics['weighted avg']['precision']:.4f}")
        logger.info(f"Recall: {best_metrics['weighted avg']['recall']:.4f}")
        logger.info(f"F1-Score: {best_metrics['weighted avg']['f1-score']:.4f}")
    
    def main(self):
        """
        Main method to run evaluation for all models.
        """
        models = self.load_models()
        
        if not models:
            logger.error("No models found for evaluation. Please train models first.")
            return
        
        test_loader, test_dataset = self.get_test_data_loader()
        results = self.evaluate_models(models, test_loader)
        self.save_results(results, test_dataset)
