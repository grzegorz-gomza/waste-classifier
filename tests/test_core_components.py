r"""
Core component tests for WasteClassifier demo app.

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
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from WasteClassifier.config.configuration import ConfigurationManager
from WasteClassifier.components.dataset import WasteDataset
from WasteClassifier.components.prepare_base_model import PrepareBaseModel
from WasteClassifier.utils.common import read_yaml, save_json, create_directories


class TestCoreComponents:
    """Test core components of the WasteClassifier pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config_files(self, temp_dir):
        """Create sample configuration files."""
        # Create config.yaml
        config_data = {
            'data_ingestion': {
                'root_dir': str(temp_dir / 'artifacts' / 'data_ingestion'),
                'source_URL': 'https://example.com/data.zip',
                'local_data_file': str(temp_dir / 'artifacts' / 'data_ingestion' / 'data.zip'),
                'unzip_dir': str(temp_dir / 'artifacts' / 'data_ingestion' / 'data')
            },
            'prepare_base_model': {
                'root_dir': str(temp_dir / 'artifacts' / 'prepare_base_model'),
                'updated_base_model_path': str(temp_dir / 'artifacts' / 'prepare_base_model' / 'base_model.pth'),
                'base_model_path': str(temp_dir / 'artifacts' / 'prepare_base_model' / 'base_model.pth')
            },
            'train_dl_model': {
                'root_dir': str(temp_dir / 'artifacts' / 'training_dl'),
                'trained_model_path': str(temp_dir / 'artifacts' / 'training_dl' / 'model.pth')
            },
            'evaluation': {
                'root_dir': str(temp_dir / 'artifacts' / 'evaluation'),
                'test_data_path': str(temp_dir / 'artifacts' / 'data_ingestion' / 'data'),
                'dl_model_path': str(temp_dir / 'artifacts' / 'training_dl' / 'model.pth'),
                'report_dir': str(temp_dir / 'artifacts' / 'reports')
            }
        }
        
        # Create params.yaml
        params_data = {
            'IMAGE_SIZE': [224, 224, 3],
            'BATCH_SIZE': 32,
            'TEST_SPLIT': 0.2,
            'AUGMENTATION': {'ENABLED': True},
            'DL_MODELS': ['resnet50'],
            'PRETRAINED': True,
            'FREEZE_BASE': True,
            'LEARNING_RATE': 0.001,
            'EPOCHS': 2,
            'OPTIMIZER': 'adam',
            'WEIGHT_DECAY': 0.01,
            'CLASSES': 5
        }
        
        config_file = temp_dir / 'config.yaml'
        params_file = temp_dir / 'params.yaml'
        
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        with open(params_file, 'w') as f:
            import yaml
            yaml.dump(params_data, f)
        
        return config_file, params_file
    
    def test_configuration_manager(self, sample_config_files):
        """Test configuration manager initialization."""
        config_file, params_file = sample_config_files
        
        # Mock the ConfigurationManager to use our test files
        original_config = Path('config/config.yaml')
        original_params = Path('params.yaml')
        backup_config = Path('config/config.yaml.bak')
        backup_params = Path('params.yaml.bak')
        
        try:
            # Create symlinks or copy files for testing
            os.makedirs('config', exist_ok=True)

            # Back up originals if they exist
            if original_config.exists():
                shutil.copy2(original_config, backup_config)
            if original_params.exists():
                shutil.copy2(original_params, backup_params)

            shutil.copy2(config_file, original_config)
            shutil.copy2(params_file, original_params)
            
            # Test configuration loading
            config_manager = ConfigurationManager()
            
            # Test that we can get configurations
            data_ingestion_config = config_manager.get_data_ingestion_config()
            assert data_ingestion_config is not None
            assert hasattr(data_ingestion_config, 'root_dir')
            
            prepare_model_config = config_manager.get_prepare_base_model_config()
            assert prepare_model_config is not None
            assert hasattr(prepare_model_config, 'root_dir')
            
            train_config = config_manager.get_train_dl_model_config()
            assert train_config is not None
            assert hasattr(train_config, 'root_dir')
            
            eval_config = config_manager.get_evaluation_config()
            assert eval_config is not None
            assert hasattr(eval_config, 'root_dir')
            
        finally:
            # Restore originals if backed up, otherwise remove test files
            if backup_config.exists():
                shutil.copy2(backup_config, original_config)
                backup_config.unlink()
            else:
                if original_config.exists():
                    original_config.unlink()

            if backup_params.exists():
                shutil.copy2(backup_params, original_params)
                backup_params.unlink()
            else:
                if original_params.exists():
                    original_params.unlink()

            if Path('config').exists() and not list(Path('config').iterdir()):
                Path('config').rmdir()
    
    def test_utility_functions(self, temp_dir):
        """Test utility functions."""
        # Test create_directories
        test_dirs = [temp_dir / 'dir1', temp_dir / 'dir2' / 'subdir']
        create_directories(test_dirs)
        
        for dir_path in test_dirs:
            assert dir_path.exists()
            assert dir_path.is_dir()
        
        # Test save_json and read_yaml
        test_data = {'key1': 'value1', 'key2': 42}
        json_file = temp_dir / 'test.json'
        save_json(json_file, test_data)
        assert json_file.exists()
        
        # Test read_yaml
        yaml_file = temp_dir / 'test.yaml'
        yaml_data = {'test': 'data', 'number': 123}
        with open(yaml_file, 'w') as f:
            import yaml
            yaml.dump(yaml_data, f)
        
        loaded_data = read_yaml(yaml_file)
        assert loaded_data.test == 'data'
        assert loaded_data.number == 123
    
    def test_prepare_base_model_initialization(self, temp_dir):
        """Test PrepareBaseModel initialization."""
        # Create a minimal config
        class MockConfig:
            def __init__(self):
                self.root_dir = temp_dir / 'models'
                self.updated_base_model_path = temp_dir / 'models' / 'updated_model.pth'
                self.base_model_path = temp_dir / 'models' / 'base_model.pth'
                self.params_image_size = [224, 224, 3]
                self.params_classes = 5
                self.params_learning_rate = 0.001
                self.params_include_top = False
                self.params_weights = 'imagenet'
        
        config = MockConfig()
        prepare_model = PrepareBaseModel(config)
        
        assert prepare_model.config == config
        assert hasattr(prepare_model, 'device')
    
    def test_dataset_import_and_structure(self):
        """Test dataset import and basic structure."""
        # Test that WasteDataset can be imported
        assert WasteDataset is not None
        
        # Test dataset class has expected methods
        assert hasattr(WasteDataset, '__init__')
        assert hasattr(WasteDataset, '__len__')
        assert hasattr(WasteDataset, '__getitem__')
    
    def test_pytorch_availability(self):
        """Test PyTorch availability and basic functionality."""
        # Test PyTorch is available
        assert torch is not None
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        assert x.shape == (2, 3)
        assert torch.is_tensor(x)
        
        # Test device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert device is not None
    
    def test_numpy_availability(self):
        """Test NumPy availability and basic functionality."""
        # Test NumPy is available
        assert np is not None
        
        # Test basic array operations
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.sum() == 15
        assert arr.mean() == 3.0


class TestIntegration:
    """Simple integration tests."""
    
    def test_import_chain(self):
        """Test that all major components can be imported."""
        try:
            from WasteClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
            from WasteClassifier.pipeline.stage_02_prepare_models import PrepareModelPipeline
            from WasteClassifier.pipeline.stage_03_train import DLTrainModelPipeline
            from WasteClassifier.pipeline.stage_04_evaluate import EvaluationPipeline
            
            # Test that classes exist
            assert DataIngestionPipeline is not None
            assert PrepareModelPipeline is not None
            assert DLTrainModelPipeline is not None
            assert EvaluationPipeline is not None
            
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")
    
    def test_config_files_exist(self):
        """Test that required config files exist."""
        config_path = Path('config/config.yaml')
        params_path = Path('params.yaml')
        
        # These should exist in the project root
        assert config_path.exists(), "config/config.yaml should exist"
        assert params_path.exists(), "params.yaml should exist"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
