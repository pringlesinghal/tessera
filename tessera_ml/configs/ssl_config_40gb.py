# configs/ssl_config_40gb.py
# Configuration optimized for 40GB GPUs (A100 40GB, A40, etc.)
# This is a reduced-memory version of ssl_config.py

# Import and modify the base config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ssl_config import config as base_config

# Create a copy and override memory-sensitive parameters
config = {**base_config}

# ========== Reduced for 40GB GPU ==========
config["batch_size"] = 512  # Reduced from 1024 for 40GB GPUs
config["cache_size"] = 50   # Reduced from 100
config["num_workers"] = 1   # Reduced from 2

# Note: Effective batch size with 8 GPUs = 512 * 8 = 4096 (still large!)

