"""
Deep learning model training component using PyTorch.

Author: Grzegorz Gomza
Date: February 2026
References:
- PyTorch training loop: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- Data augmentation: https://pytorch.org/vision/stable/transforms.html
"""

import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from textwrap import dedent

from WasteClassifier.entity.config_entity import TrainDLModelConfig, TrackingConfig, VisualizationConfig
from WasteClassifier.components.share.dataset import WasteDataset
from WasteClassifier.components.visualization.plot_artifacts import PlotArtifacts, PlotConfig
from WasteClassifier.utils.mlflow_utils import (
    make_run_name,
    log_artifacts_dir,
    log_metrics,
    log_params,
    start_run,
    write_run_context,
)

logger = logging.getLogger(__name__)

class TrainDLModel:
    def __init__(
        self,
        config: TrainDLModelConfig,
        tracking_config: TrackingConfig,
        visualization_config: VisualizationConfig,
    ):
        """
        Initialize TrainDLModel component.
        
        Args:
            config (TrainDLModelConfig): Training configuration
        """
        self.config = config
        self.tracking_config = tracking_config
        self.visualization_config = visualization_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def get_transforms(self):
        """
        Get data transforms for training and validation.
        
        Returns:
            tuple: (train_transform, val_transform)
        """
        image_size = tuple(self.config.params_image_size[:2])
        
        if self.config.params_augmentation:
            train_transform = transforms.Compose([
                transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                # Normalize with ImageNet statistics
                # Resource: https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # Normalize with ImageNet statistics
                # Resource: https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # Normalize with ImageNet statistics
            # Resource: https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def get_data_loaders(self):
        """
        Create data loaders for training and validation.
        Validation set is created from the training subset.
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        train_transform, val_transform = self.get_transforms()
        
        # Get full training dataset
        full_train_dataset = WasteDataset(
            root_dir=self.config.training_data,
            transform=None,  # Don't apply transforms yet
            split='train',
            test_split=self.config.params_test_split
        )
        
        # Create validation set from training subset (same ratio as test split)
        val_size = int(self.config.params_test_split * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Apply transforms to the split datasets
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self):
        """
        Train all deep learning models.
        """
        logger.info("Starting training for all models...")

        # Log training parameters
        logger.info("Training Parameters:")
        logger.info(f"  Epochs: {self.config.params_epochs}")
        logger.info(f"  Batch Size: {self.config.params_batch_size}")
        logger.info(f"  Learning Rate: {self.config.params_learning_rate}")
        logger.info(f"  Optimizer: {self.config.params_optimizer}")
        logger.info(f"  Weight Decay: {self.config.params_weight_decay}")
        logger.info(f"  Augmentation: {self.config.params_augmentation}")
        logger.info(f"  Test Split: {self.config.params_test_split}")
        logger.info(f"  Image Size: {self.config.params_image_size}")
        logger.info(f"  Models: {self.config.params_models}")

        # Prepare data (shared across all models)
        train_loader, val_loader = self.get_data_loaders()
        
        # Train each model
        model_results = {}
        
        for model_name in self.config.params_models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model: {model_name}")
            logger.info(f"{'='*60}")
            
            # Per-model MLflow run
            run_dt = datetime.now()
            run_name = make_run_name("DL", model_name, run_dt)
            # Stable per-model reports directory consumed by the app.
            report_dir = Path("artifacts") / "reports" / model_name
            report_dir.mkdir(parents=True, exist_ok=True)

            if self.tracking_config.enabled:
                run = start_run(
                    run_name=run_name,
                    tracking_uri=self.tracking_config.tracking_uri,
                    experiment_name=self.tracking_config.experiment_name,
                    tags={
                        "model_type": "DL",
                        "model_name": model_name,
                        "run_name": run_name,
                    },
                )
                log_params(
                    {
                        "model_type": "DL",
                        "model_name": model_name,
                        "epochs": self.config.params_epochs,
                        "batch_size": self.config.params_batch_size,
                        "learning_rate": self.config.params_learning_rate,
                        "optimizer": self.config.params_optimizer,
                        "weight_decay": self.config.params_weight_decay,
                        "augmentation": self.config.params_augmentation,
                        "test_split": self.config.params_test_split,
                        "image_size": self.config.params_image_size,
                    }
                )
                run_id = run.info.run_id
            else:
                run_id = ""

            # Load model
            model_path = self.config.updated_base_model_path.parent / f"{model_name}_model.pth"
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                continue
                
            model = torch.load(model_path, weights_only=False)
            model = model.to(self.device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            
            # Choose optimizer based on configuration
            if self.config.params_optimizer.lower() == 'adamw':
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=self.config.params_learning_rate,
                    weight_decay=self.config.params_weight_decay
                )
            else:
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=self.config.params_learning_rate
                )
            # Scheduler for learning rate reduction on plateau
            # Resource: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3
            )

            best_val_acc = 0.0
            model_metrics = {
                'train_losses': [],
                'train_accs': [],
                'val_losses': [],
                'val_accs': []
            }

            for epoch in range(self.config.params_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                train_bar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{self.config.params_epochs} [Train]")
                for images, labels in train_bar:
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                    train_bar.set_postfix({
                        'loss': f'{train_loss/len(train_loader):.3f}',
                        'acc': f'{100.*train_correct/train_total:.2f}%'
                    })

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    val_bar = tqdm(val_loader, desc=f"{model_name} Epoch {epoch+1}/{self.config.params_epochs} [Val]")
                    for images, labels in val_bar:
                        images, labels = images.to(self.device), labels.to(self.device)

                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                        val_bar.set_postfix({
                            'loss': f'{val_loss/len(val_loader):.3f}',
                            'acc': f'{100.*val_correct/val_total:.2f}%'
                        })

                train_acc = 100. * train_correct / train_total
                val_acc = 100. * val_correct / val_total
                
                # Store metrics
                model_metrics['train_losses'].append(train_loss / len(train_loader))
                model_metrics['train_accs'].append(train_acc)
                model_metrics['val_losses'].append(val_loss / len(val_loader))
                model_metrics['val_accs'].append(val_acc)

                logger.info(f"{model_name} Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.3f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.3f}, Val Acc: {val_acc:.2f}%")

                if self.tracking_config.enabled:
                    log_metrics(
                        {
                            "train_loss": train_loss / len(train_loader),
                            "train_acc": train_acc / 100.0,
                            "val_loss": val_loss / len(val_loader),
                            "val_acc": val_acc / 100.0,
                        },
                        step=epoch,
                    )

                # Learning rate scheduling
                scheduler.step(val_acc)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = self.config.root_dir / f"{model_name}_best_model.pth"
                    torch.save(model, best_model_path)
                    logger.info(f"Best {model_name} model saved with validation accuracy: {val_acc:.2f}%")

            model_results[model_name] = {
                'best_val_acc': best_val_acc,
                'metrics': model_metrics
            }

            logger.info(f"{model_name} training completed. Best validation accuracy: {best_val_acc:.2f}%")

            # Persist curves for later aggregation/comparison
            try:
                curves_path = report_dir / "training_curves.json"
                curves_path.write_text(
                    __import__("json").dumps(
                        {
                            "model_name": model_name,
                            "train_losses": model_metrics["train_losses"],
                            "train_accs": model_metrics["train_accs"],
                            "val_losses": model_metrics["val_losses"],
                            "val_accs": model_metrics["val_accs"],
                        },
                        indent=2,
                    )
                )
            except Exception:
                pass

            # Plot + log artifacts (optional)
            plotter = PlotArtifacts(
                report_dir,
                config=PlotConfig(enabled=self.visualization_config.enabled, dpi=self.visualization_config.dpi),
            )
            plotter.plot_dl_training_progress(
                model_name=model_name,
                train_losses=model_metrics["train_losses"],
                train_accs=model_metrics["train_accs"],
                val_losses=model_metrics["val_losses"],
                val_accs=model_metrics["val_accs"],
                run_name=run_name,
                output_name="training_progress.png",
            )

            if self.tracking_config.enabled:
                best_model_path = self.config.root_dir / f"{model_name}_best_model.pth"
                if best_model_path.exists():
                    best_model = torch.load(best_model_path, weights_only=False)
                    best_model = best_model.to(self.device)

                    import mlflow
                    import mlflow.pytorch

                    mlflow.pytorch.log_model(best_model, artifact_path="model")
                    try:
                        mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")
                    except Exception:
                        pass

                log_artifacts_dir(report_dir, artifact_path="reports")

                context_path = self.tracking_config.runs_root_dir / run_name / "run_context.json"
                write_run_context(
                    context_path=context_path,
                    run_name=run_name,
                    run_id=run_id,
                    model_type="DL",
                    model_name=model_name,
                    report_dir=report_dir,
                )

                try:
                    import mlflow

                    mlflow.end_run()
                except Exception:
                    pass

        # Log model comparison
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info(f"{'='*60}")
        
        for model_name, results in model_results.items():
            logger.info(f"{model_name}: {results['best_val_acc']:.2f}% validation accuracy")
        
        # Find best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['best_val_acc'])
        logger.info(f"\n🏆 Best performing model: {best_model_name} with {model_results[best_model_name]['best_val_acc']:.2f}% accuracy")
        
        # Copy best model to default location for backward compatibility
        best_model_path = self.config.root_dir / f"{best_model_name}_best_model.pth"
        if best_model_path.exists():
            torch.save(torch.load(best_model_path, weights_only=False), self.config.trained_model_path)
            logger.info(f"Best model copied to {self.config.trained_model_path}")
    
    def main(self):
        """
        Main method to run training.
        """
        self.train()
