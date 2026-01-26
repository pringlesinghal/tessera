# Fast batch size test config - minimal dataset loading
config = {
    # Use smaller dataset 
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled",
    "data_root": "/scratch/users/psinghal/time_series", 
    "years": [2016],  # Just one year for speed
    "cache_size": 10,  # Small cache
    
    # Test batch size
    "batch_size": 1024,  # Target: 4096 global batch
    "epochs": 1,
    
    # Minimal training params
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "warmup_ratio": 0.1,
    "weight_decay": 1e-6,
    "clip_grad_norm": 2.0,
    
    # Model - same as working config
    "fusion_method": "concat",
    "latent_dim": 128,
    "s2_num_heads": 4,
    "s2_num_layers": 4, 
    "s2_dim_feedforward": 4096,
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 4096,
    "projector_out_dim": 4096,
    "projector_hidden_dim": 4096,
    
    # Data params
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "num_workers": 2,
    
    # Fast logging
    "log_interval_steps": 1,
    "val_interval_steps": 0,  # No validation
    
    # Optimizations
    "apply_amp": True,  # Essential for large batch
    "use_torch_compile": False,
    "apply_mixup": False,  # Disable for speed
    
    # QAT disabled
    "apply_qat_representation": False,
    "qat_representation_start_step": 999999,
    
    # No resume
    "resume_from_checkpoint": None,
    
    # Wandb
    "wandb_project": "batch-test-1024",
    "wandb_api_key": "",
    "disable_wandb_git": True,
}