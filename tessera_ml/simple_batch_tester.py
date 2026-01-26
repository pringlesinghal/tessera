#!/usr/bin/env python3
"""
Simple Batch Size Testing Script for 4 A100 GPUs
Tests a series of batch sizes directly with torchrun (assumes running inside container)
"""

import os
import time
import subprocess
import json
import torch
import torch.distributed as dist
from pathlib import Path
import sys

def create_test_config(batch_size, enable_amp=True):
    """Create a test configuration for batch size testing"""
    config = {
        # TreeDataset Configuration
        "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled",
        "data_root": "/scratch/users/psinghal/time_series", 
        "years": [2016, 2017],  # Limited years for faster testing
        "cache_size": 20,
        
        # Training Parameters - Test Configuration
        "batch_size": batch_size,
        "epochs": 1,
        "learning_rate": 0.002,
        "barlow_lambda": 5e-3,
        "warmup_ratio": 0.1,
        "plateau_ratio": 0,
        "weight_decay": 1e-6,
        "clip_grad_norm": 2.0,
        
        # Model Architecture - Conservative for memory testing
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
        
        # Data Parameters
        "sample_size_s2": 40,
        "sample_size_s1": 40,
        "num_workers": 2,
        "shuffle_tiles": True,
        
        # Logging - Minimal for testing
        "log_interval_steps": 2,  # Log every 2 steps
        "val_interval_steps": 0,  # Disable validation during testing
        
        # Augmentation
        "apply_mixup": True,
        "mixup_lambda": 1.0,
        "beta_alpha": 1.0,
        "beta_beta": 1.0,
        
        # Performance Optimization
        "apply_amp": enable_amp,
        "use_torch_compile": False,
        
        # QAT - Disabled for testing
        "apply_qat_representation": False,
        "qat_representation_start_step": 999999,  # Large number instead of inf
        
        # Resume Training
        "resume_from_checkpoint": None,
        
        # Wandb
        "wandb_project": f"batch-test-{batch_size}",
        "wandb_api_key": "",
        "disable_wandb_git": True,
    }
    return config

def test_batch_size(batch_size, num_gpus=4, max_steps=3, enable_amp=True):
    """Test a specific batch size"""
    print(f"\n{'='*60}")
    print(f"Testing: batch_size={batch_size}, global_batch_size={batch_size * num_gpus}, AMP={enable_amp}")
    print(f"{'='*60}")
    
    # Create config file
    config = create_test_config(batch_size, enable_amp)
    config_file = f"test_config_{batch_size}_{int(enable_amp)}.py"
    
    with open(config_file, 'w') as f:
        f.write(f"config = {repr(config)}\n")
    
    # Set environment
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["NCCL_DEBUG"] = "WARN"
    
    # Command to run
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "train_ssl_multi_gpu.py",
        "--config", config_file
    ]
    
    success = False
    step_count = 0
    peak_memory_gb = 0
    start_time = time.time()
    
    try:
        print("Starting training process...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor output
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                print(line)
                
                # Count steps
                if "Step=" in line and "Loss=" in line:
                    step_count += 1
                    print(f"   -> Completed step {step_count}/{max_steps}")
                    
                    if step_count >= max_steps:
                        print(f"   -> Reached {max_steps} steps, terminating...")
                        process.terminate()
                        success = True
                        break
                
                # Look for OOM or other errors
                if any(error in line.lower() for error in ['out of memory', 'cuda error', 'runtime error']):
                    print(f"   -> ERROR DETECTED: {line}")
                    process.terminate()
                    break
                
                # Track GPU memory
                if "GPU memory usage" in line:
                    try:
                        memory_str = line.split("GPU memory usage")[1].split("MB")[0]
                        memory_mb = float(memory_str.replace(":", "").strip())
                        memory_gb = memory_mb / 1024
                        peak_memory_gb = max(peak_memory_gb, memory_gb)
                    except:
                        pass
        
        # Wait for process to finish
        process.wait()
        if step_count >= max_steps:
            success = True
            
    except Exception as e:
        print(f"   -> Exception: {e}")
        success = False
    
    # Cleanup
    try:
        os.remove(config_file)
    except:
        pass
    
    duration = time.time() - start_time
    
    result = {
        'batch_size': batch_size,
        'global_batch_size': batch_size * num_gpus,
        'success': success,
        'steps_completed': step_count,
        'duration_sec': duration,
        'peak_memory_gb': peak_memory_gb,
        'enable_amp': enable_amp
    }
    
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"   -> Result: {status} ({step_count} steps, {peak_memory_gb:.1f}GB peak)")
    
    return result

def main():
    print("🚀 Batch Size Testing for 4 A100 GPUs")
    print("Target: Find optimal batch size for global batch ≥ 4096")
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    results = []
    
    # Test sequence: start small, work up
    test_cases = [
        (2, False),    # Your current working config
        (4, False),    # Double it
        (8, True),     # Enable AMP, quadruple
        (16, True),    # 
        (32, True),    # 
        (64, True),    # 
        (128, True),   # 
        (256, True),   # 
        (512, True),   # Global = 2048
        (1024, True),  # Global = 4096 - TARGET!
        (2048, True),  # Global = 8192 - stretch goal
    ]
    
    print(f"\nTesting {len(test_cases)} configurations...\n")
    
    successful_configs = []
    
    for batch_size, amp in test_cases:
        result = test_batch_size(batch_size, enable_amp=amp)
        results.append(result)
        
        if result['success']:
            successful_configs.append(result)
            print(f"✅ Success: {batch_size} per GPU = {result['global_batch_size']} global batch")
        else:
            print(f"❌ Failed: {batch_size} per GPU")
            # If we fail with AMP, maybe stop testing larger sizes
            if amp and batch_size >= 64:
                print("   -> Skipping larger batch sizes due to failure...")
                break
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH SIZE TEST RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Batch/GPU':<10} {'Global':<8} {'AMP':<5} {'Steps':<6} {'Memory':<8} {'Status':<8}")
    print(f"{'-'*10} {'-'*8} {'-'*5} {'-'*6} {'-'*8} {'-'*8}")
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{r['batch_size']:<10} {r['global_batch_size']:<8} {r['enable_amp']!s:<5} "
              f"{r['steps_completed']:<6} {r['peak_memory_gb']:<8.1f} {status:<8}")
    
    # Find optimal config
    target_configs = [r for r in successful_configs if r['global_batch_size'] >= 4096]
    
    if target_configs:
        best = max(target_configs, key=lambda x: x['global_batch_size'])
        print(f"\n🎯 OPTIMAL CONFIG FOUND:")
        print(f"   Batch size per GPU: {best['batch_size']}")
        print(f"   Global batch size: {best['global_batch_size']}")
        print(f"   AMP enabled: {best['enable_amp']}")
        print(f"   Peak memory: {best['peak_memory_gb']:.1f} GB per GPU")
    elif successful_configs:
        best = max(successful_configs, key=lambda x: x['global_batch_size'])
        print(f"\n⚠️ TARGET NOT REACHED. Best found:")
        print(f"   Batch size per GPU: {best['batch_size']}")
        print(f"   Global batch size: {best['global_batch_size']}")
        print(f"   AMP enabled: {best['enable_amp']}")
        print(f"   Peak memory: {best['peak_memory_gb']:.1f} GB per GPU")
    else:
        print("\n❌ No successful configurations!")
    
    # Save results
    with open('batch_size_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to batch_size_results.json")

if __name__ == "__main__":
    main()