import os
import sys

# Add the project root to the Python path (two levels up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
print(f"Added path: {project_root}")

try:
    import source.data.processing
    print("Successfully imported source.data.processing")
except ImportError as e:
    print(f"Failed to import source.data.processing: {e}")

try:
    from tsunami.data.processing import DataProcessor
    print("Successfully imported DataProcessor from tsunami.data.processing")
except ImportError as e:
    print(f"Failed to import DataProcessor: {e}")
