# Optimized config for 4 A100 GPUs - fixes socket issues and maximizes batch size
config = {
    # ========== TreeDataset Configuration ==========
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled",  # Smaller dataset
    "data_root": "/scratch/users/psinghal/time_series", 
    "years": [2016, 2017],  # Just 2 years for faster loading
    "cache_size": 50,  # Higher cache for A100s
    
    # ========== Training Parameters - Target 4096 Global Batch ==========
    "batch_size": 1024,  # 1024 × 4 = 4096 global batch size
    "epochs": 100,       # Full training
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "warmup_ratio": 0.1,
    "plateau_ratio": 0.7,
    "weight_decay": 1e-6,
    "clip_grad_norm": 2.0,
    
    # ========== Model Architecture - Proven Working Config ==========
    "fusion_method": "concat",
    "latent_dim": 128,
    # S2 encoder
    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 4096,
    # S1 encoder
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 4096,
    # Projection head
    "projector_out_dim": 4096,
    "projector_hidden_dim": 4096,
    
    # ========== Data Parameters ==========
    "sample_size_s2": 40,
    "sample_size_s1": 40,
    "num_workers": 4,  # More workers for A100s
    "shuffle_tiles": True,
    
    # ========== Logging and Validation ==========
    "log_interval_steps": 50,   # Less frequent for large batches
    "val_interval_steps": 1000,
    "eval_method": "linear_probe",
    
    # Validation data paths
    "val_s2_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/bands_downsample_100.npy",
    "val_s2_masks_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/masks_downsample_100.npy",
    "val_s2_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/doys.npy",
    "val_s1_asc_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_ascending_downsample_100.npy",
    "val_s1_asc_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_ascending_doy.npy",
    "val_s1_desc_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_descending_downsample_100.npy",
    "val_s1_desc_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_descending_doy.npy",
    "val_labels_path": "data/ssl_training/austrian_crop_v1.0_pipeline/fieldtype_17classes_downsample_100.npy",
    "field_id_path": "data/ssl_training/austrian_crop_v1.0_pipeline/fieldid_downsample_100.npy",
    "fielddata_csv_path": "data/ssl_training/austrian_crop_v1.0_pipeline/updated_fielddata.csv",
    "training_ratio": 0.1,
    "val_test_split_ratio": 1 / 7,
    "num_inference": 1,
    "classifier_type": "lr",
    "val_standardize": True,
    "val_min_valid_timesteps": 0,
    "val_batch_size": 512,
    "val_num_workers": 2,
    
    # ========== Augmentation ==========
    "apply_mixup": True,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    
    # ========== Dataset Size ==========
    "total_samples": None,  # Auto-calculate
    
    # ========== Performance Optimization - CRITICAL FOR MEMORY ==========
    "apply_amp": True,      # ESSENTIAL for large batches
    "use_torch_compile": False,  # Keep disabled in apptainer
    
    # ========== Quantization Aware Training ==========
    "apply_qat_representation": False,  # Disabled for max performance
    "qat_representation_bits": 8,
    "qat_representation_symmetric": True,
    "qat_representation_start_step": 999999,
    
    # ========== Resume Training ==========
    "resume_from_checkpoint": None,
    "resume_learning_rate": 0.002,
    "resume_warmup_steps": 0,
    
    # ========== Wandb Configuration ==========
    "wandb_project": "tessera-ssl-4gpu-optimized",
    "wandb_api_key": "",
    "wandb_watch_freq": 400,
    "disable_wandb_git": True,
}