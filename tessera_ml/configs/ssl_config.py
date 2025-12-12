# configs/ssl_config.py
# Configuration for train_ssl_multi_gpu.py with TreeDataset
# Adapted from tessera-util/configs/ssl_config.py for FSDP training

config = {
    # ========== TreeDataset Configuration (NEW - REQUIRED) ==========
    # UPDATE THESE PATHS to your actual data locations
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled",
    "data_root": "/scratch/users/psinghal/time_series",
    "years": [
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
    ],  # 2016-2024 inclusive
    "cache_size": 100,  # LRU cache size for tiles (increased for multi-GPU training)
    # ========== Training Parameters ==========
    "batch_size": 2048,  # Per GPU batch size
    "epochs": 1,
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "warmup_ratio": 0.1,
    "plateau_ratio": 0,
    "weight_decay": 1e-6,
    "clip_grad_norm": 2.0,
    # ========== Model Architecture ==========
    "fusion_method": "concat",  # Options: 'sum', 'concat', 'transformer'
    "latent_dim": 128,
    # S2 (Sentinel-2) encoder
    "s2_num_heads": 4,
    "s2_num_layers": 4,
    "s2_dim_feedforward": 4096,
    # S1 (Sentinel-1) encoder
    "s1_num_heads": 4,
    "s1_num_layers": 4,
    "s1_dim_feedforward": 4096,
    # Projection head
    "projector_out_dim": 8192 * 2,
    "projector_hidden_dim": 8192 * 2,
    # ========== Data Parameters ==========
    "sample_size_s2": 40,  # Sentinel-2 time steps
    "sample_size_s1": 40,  # Sentinel-1 time steps
    "num_workers": 2,  # DataLoader workers per GPU
    "shuffle_tiles": True,
    # ========== Logging and Validation ==========
    "log_interval_steps": 10,
    "val_interval_steps": 600,  # Set to 0 to disable validation
    "eval_method": "linear_probe",
    # Validation data paths (Austrian Crop dataset)
    "val_s2_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/bands_downsample_100.npy",
    "val_s2_masks_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/masks_downsample_100.npy",
    "val_s2_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/doys.npy",
    "val_s1_asc_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_ascending_downsample_100.npy",
    "val_s1_asc_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_ascending_doy.npy",
    "val_s1_desc_bands_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_descending_downsample_100.npy",
    "val_s1_desc_doy_file_path": "data/ssl_training/austrian_crop_v1.0_pipeline/sar_descending_doy.npy",
    "val_labels_path": "data/ssl_training/austrian_crop_v1.0_pipeline/fieldtype_17classes_downsample_100.npy",
    # Field-based validation parameters
    "field_id_path": "data/ssl_training/austrian_crop_v1.0_pipeline/fieldid_downsample_100.npy",
    "fielddata_csv_path": "data/ssl_training/austrian_crop_v1.0_pipeline/updated_fielddata.csv",
    "training_ratio": 0.1,  # Proportion of field area for training
    "val_test_split_ratio": 1 / 7,  # Split between validation and test sets
    "num_inference": 1,  # Number of inference passes (averaging)
    "classifier_type": "lr",  # Options: 'lr' (LogisticRegression) or 'rf' (RandomForest)
    "val_standardize": True,
    "val_min_valid_timesteps": 0,
    "val_batch_size": 512,
    "val_num_workers": 0,
    # ========== Augmentation ==========
    "apply_mixup": True,
    "mixup_lambda": 1.0,
    "beta_alpha": 1.0,
    "beta_beta": 1.0,
    # ========== Performance Optimization ==========
    "apply_amp": False,  # Mixed precision training (set to True for faster training)
    "use_torch_compile": True,  # PyTorch 2.0+ compile (may not work with FSDP)
    # ========== Quantization Aware Training (QAT) ==========
    "apply_qat_representation": True,  # Enable QAT for representation
    "qat_representation_bits": 8,  # Quantization bits (typically 8)
    "qat_representation_symmetric": True,
    "qat_representation_start_step": 5000,  # Start QAT after this many steps
    # ========== Resume Training ==========
    "resume_from_checkpoint": None,  # Set to checkpoint path to resume
    # "resume_from_checkpoint": "checkpoints/ssl/checkpoint_20250605_121934.pt",
    "resume_learning_rate": 0.002,
    "resume_warmup_steps": 0,
    # ========== Wandb Configuration ==========
    "wandb_project": "tessera-ssl-fsdp",
    "wandb_api_key": "",  # Add your W&B API key here or set WANDB_API_KEY env var
    "wandb_watch_freq": 400,
    "disable_wandb_git": True,
}
