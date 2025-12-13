#!/usr/bin/env python3
"""
Script to add FSDP functionality to the working DDP training script
Run this after DDP tests pass successfully
"""

def add_fsdp_to_training_script():
    """
    Modifies train_multi_gpu_v2.py to add FSDP support
    This is separated so we can test DDP first, then add FSDP incrementally
    """
    
    script_path = "tessera_ml/train_multi_gpu_v2.py"
    
    # Read current script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if already has FSDP
    if "FSDP(" in content and "# FSDP INTEGRATION ADDED" in content:
        print("FSDP already integrated in the script")
        return
    
    # Add FSDP integration
    fsdp_integration = '''
    # === FSDP INTEGRATION ADDED ===
    # Replace DDP with FSDP after DDP testing is successful
    
    # FSDP configuration
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        cast_forward_inputs=False,
        cast_root_forward_inputs=True,
        _module_classes_to_ignore=(torch.nn.BatchNorm1d,)
    )
    
    # Custom auto wrap policy
    def custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
        return recurse
    
    fsdp_config_dict = {
        "auto_wrap_policy": custom_auto_wrap_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": device,
        "sync_module_states": True,
        "forward_prefetch": True,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "cpu_offload": CPUOffload(offload_params=False),
        "use_orig_params": True,
    }
    if mixed_precision_policy:
        fsdp_config_dict["mixed_precision"] = mixed_precision_policy

    if global_rank == 0:
        logging.info("Wrapping model with FSDP")
        logging.info(f"FSDP Configuration: {json.dumps({k: str(v) for k, v in fsdp_config_dict.items()}, indent=2)}")

    model = FSDP(model, **fsdp_config_dict)
    '''
    
    # Replace DDP section with FSDP
    ddp_section = "    # === START WITH BASIC DDP (not FSDP yet) ===\n    # For now, let's use standard DDP to verify everything works\n    from torch.nn.parallel import DistributedDataParallel as DDP\n    \n    if global_rank == 0:\n        logging.info(\"Wrapping model with DDP (testing phase)\")\n    \n    model = DDP(model, device_ids=[local_rank])"
    
    new_content = content.replace(ddp_section, fsdp_integration)
    
    # Backup original
    import shutil
    shutil.copy(script_path, f"{script_path}.backup")
    
    # Write new version
    with open(script_path, 'w') as f:
        f.write(new_content)
    
    print("✅ FSDP integration added to train_multi_gpu_v2.py")
    print("Backup saved as train_multi_gpu_v2.py.backup")

if __name__ == "__main__":
    print("Adding FSDP integration to training script...")
    add_fsdp_to_training_script()
    print("Done! Now test with the same commands as before.")