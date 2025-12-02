import sys
import os

# Add the parent directory of tessera_ml to sys.path
# Assuming we run this from tessera/tessera_ml
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")
print(f"Sys path: {sys.path}")

print("Checking imports...")
try:
    from tessera_ml.dataset import TreeDataset
    print("TreeDataset imported.")
    from tessera_ml.models.ssl_model import MultimodalBTModel
    print("MultimodalBTModel imported.")
    # from tessera_ml.train_ssl import main
    # print("train_ssl imported.")
    print("All imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
