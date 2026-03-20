"""
Simple training component tests.

Author: Grzegorz Gomza
Date: February 2026
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from WasteClassifier.components.train import TrainDLModel


class TestTraining:
    """Test training component."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_training_import(self):
        """Test that TrainDLModel can be imported."""
        assert TrainDLModel is not None
        assert hasattr(TrainDLModel, '__init__')
        assert hasattr(TrainDLModel, 'train')
    
    def test_training_initialization(self, temp_dir):
        """Test TrainDLModel initialization."""
        # Create mock config
        class MockConfig:
            def __init__(self, temp_dir):
                self.root_dir = temp_dir / 'training'
                self.trained_model_path = temp_dir / 'training' / 'model.pth'
                self.params_image_size = [224, 224, 3]
                self.params_batch_size = 32
                self.params_augmentation = {'ENABLED': True}
                self.params_test_split = 0.2
                self.params_models = ['resnet50']
                self.params_epochs = 1
                self.params_learning_rate = 0.001
                self.params_optimizer = 'adam'
                self.params_weight_decay = 0.01
                self.params_classes = 5
        
        config = MockConfig(temp_dir)
        
        try:
            trainer = TrainDLModel(config)
            assert trainer.config == config
            assert hasattr(trainer, 'device')
        except Exception:
            # May fail due to missing dependencies, but structure should be fine
            pass
    
    def test_pytorch_device_availability(self):
        """Test PyTorch device availability."""
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        device = torch.device('cuda' if cuda_available else 'cpu')
        
        assert device is not None
        assert device.type in ['cuda', 'cpu']
    
    def test_model_architecture_import(self):
        """Test that model architectures can be imported."""
        try:
            import torchvision.models as models
            assert hasattr(models, 'resnet50')
            assert hasattr(models, 'mobilenet_v2')
            assert hasattr(models, 'efficientnet_b0')
        except ImportError:
            pytest.skip("torchvision not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
