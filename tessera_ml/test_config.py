"""
Test configuration for multi-GPU training
Smaller scale to verify functionality before full training
"""

config = {
    # Data paths (use your existing working paths)
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled",
    "data_root": "/scratch/groups/dlobell/psinghal/sentineldownloader",
    
    # Model architecture (smaller for testing)
    "latent_dim": 512,
    "projector_hidden_dim": 8192,  # Reduced from 16384
    "projector_out_dim": 8192,     # Reduced from 16384
    
    # Training parameters (smaller batch for testing) 
    "batch_size": 64,              # Reduced from 256
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 1,                   # Just 1 epoch for testing
    
    # Data parameters
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "fusion_method": "concat",
    
    # SSL parameters
    "barlow_lambda": 0.005,
    
    # Architecture details
    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 1024,
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 1024,
    
    # Quantization (disabled for testing)
    "apply_qat_representation": False,
    "qat_representation_bits": 8,
    
    # W&B settings (disabled for testing)
    "wandb_project": "tessera-multi-gpu-test",
}