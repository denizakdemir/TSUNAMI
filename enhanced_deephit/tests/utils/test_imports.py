import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
print(f"Added path: {project_root}")

try:
    import data.processing
    print("Successfully imported data.processing")
except ImportError as e:
    print(f"Failed to import data.processing: {e}")

try:
    from data.processing import DataProcessor
    print("Successfully imported DataProcessor")
except ImportError as e:
    print(f"Failed to import DataProcessor: {e}")