"""
Simple test runner for WasteClassifier demo app.

Author: Grzegorz Gomza
Date: February 2026
"""

import subprocess
import sys
from pathlib import Path


def run_test_file(test_file):
    """Run a single test file."""
    print(f"\n{'='*50}")
    print(f"Running {test_file}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            str(test_file), '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False


def main():
    """Run all simple tests."""
    test_dir = Path(__file__).parent
    
    # Simple test files to run
    test_files = [
        'test_core_components.py',
        'test_data_ingestion_simple.py', 
        'test_training_simple.py',
        'test_evaluation_simple.py'
    ]
    
    print("🧪 Running WasteClassifier Test Suite")
    print("Testing core components for demo app...")
    
    passed = 0
    total = len(test_files)
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            if run_test_file(test_path):
                passed += 1
                print(f"✅ {test_file} PASSED")
            else:
                print(f"❌ {test_file} FAILED")
        else:
            print(f"⚠️  {test_file} NOT FOUND")
            total -= 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
