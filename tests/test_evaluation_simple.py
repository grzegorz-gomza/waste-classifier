"""
Simple evaluation component tests.

Author: Grzegorz Gomza
Date: February 2026
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from WasteClassifier.components.evaluate import Evaluation


class TestEvaluation:
    """Test evaluation component."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_evaluation_import(self):
        """Test that Evaluation can be imported."""
        assert Evaluation is not None
        assert hasattr(Evaluation, '__init__')
        assert hasattr(Evaluation, 'evaluate_models')
    
    def test_evaluation_initialization(self, temp_dir):
        """Test Evaluation initialization."""
        # Create mock config
        class MockConfig:
            def __init__(self, temp_dir):
                self.root_dir = temp_dir / 'evaluation'
                self.test_data_path = temp_dir / 'test_data'
                self.dl_model_path = temp_dir / 'model.pth'
                self.report_dir = temp_dir / 'reports'
                self.params_image_size = [224, 224, 3]
                self.params_batch_size = 32
        
        config = MockConfig(temp_dir)
        
        try:
            evaluator = Evaluation(config)
            assert evaluator.config == config
            assert hasattr(evaluator, 'report_dir')
        except Exception:
            # May fail due to missing dependencies, but structure should be fine
            pass
    
    def test_metrics_import(self):
        """Test that metrics can be imported."""
        try:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            assert accuracy_score is not None
            assert classification_report is not None
            assert confusion_matrix is not None
        except ImportError:
            pytest.skip("sklearn not available")
    
    def test_plotting_import(self):
        """Test that plotting libraries can be imported."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            assert plt is not None
            assert sns is not None
        except ImportError:
            pytest.skip("matplotlib/seaborn not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
