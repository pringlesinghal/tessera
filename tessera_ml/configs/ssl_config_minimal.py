# configs/ssl_config_minimal.py
# Minimal configuration for quick testing
# Use this for debugging or single-GPU tests

config = {
    # ========== DATA PATHS (UPDATE THESE!) ==========
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled",
    "data_root": "/scratch/users/psinghal/time_series",
    "years": [2016, 2017, 2018, 2019, 2020],  # Fewer years for testing
    "cache_size": 25,  # Smaller cache for testing
    
    # ========== Training (Minimal for Testing) ==========
    "batch_size": 32,      # Small batch for testing
    "epochs": 1,           # Just 1 epoch for testing
    "learning_rate": 0.001,
    "barlow_lambda": 5e-3,
    "warmup_ratio": 0.1,
    "plateau_ratio": 0,
    "weight_decay": 1e-6,
    "clip_grad_norm": 2.0,
    
    # ========== Model Architecture (Smaller for Testing) ==========
    "fusion_method": "concat",
    "latent_dim": 128,
    
    "s2_num_heads": 4,
    "s2_num_layers": 2,  # Fewer layers for faster testing
    "s2_dim_feedforward": 1024,
    
    "s1_num_heads": 4,
    "s1_num_layers": 2,  # Fewer layers for faster testing
    "s1_dim_feedforward": 1024,
    
    "projector_out_dim": 2048,  # Smaller projection head
    "projector_hidden_dim": 2048,
    
    # ========== Data Parameters ==========
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "num_workers": 2,
    "shuffle_tiles": True,
    
    # ========== Logging (Frequent for Testing) ==========
    "log_interval_steps": 5,  # Log every 5 steps for testing
    "val_interval_steps": 0,  # Disable validation for quick testing
    
    # ========== Augmentation ==========
    "apply_mixup": False,  # Disable for faster testing
    
    # ========== Performance ==========
    "apply_amp": True,  # Enable for faster testing
    "use_torch_compile": False,  # Disable for stability
    
    # ========== QAT (Disabled for Testing) ==========
    "apply_qat_representation": False,
    
    # ========== Wandb ==========
    "wandb_project": "tessera-ssl-test",
    "wandb_api_key": "",
    "disable_wandb_git": True,
}

