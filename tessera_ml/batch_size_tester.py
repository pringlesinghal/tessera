#!/usr/bin/env python3
"""
Batch Size Testing Script for 4 A100 GPUs
Systematically tests different batch sizes to find optimal configuration for target of 4096+ batch size
"""

import os
import time
import subprocess
import json
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np

# Get current script directory
SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR / "configs"

def create_test_config(batch_size, enable_amp=True, enable_compile=False, cache_size=20):
    """Create a test configuration for batch size testing"""
    config = {
        # TreeDataset Configuration
        "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_valid_shuffled",
        "data_root": "/scratch/users/psinghal/time_series", 
        "years": [2016, 2017],  # Limited years for faster testing
        "cache_size": cache_size,
        
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
        
        # Data Parameters
        "sample_size_s2": 40,
        "sample_size_s1": 40,
        "num_workers": 2,
        "shuffle_tiles": True,
        
        # Logging - Minimal for testing
        "log_interval_steps": 1,
        "val_interval_steps": 0,  # Disable validation during testing
        "eval_method": "linear_probe",
        
        # Augmentation - Simplified for testing
        "apply_mixup": True,
        "mixup_lambda": 1.0,
        "beta_alpha": 1.0,
        "beta_beta": 1.0,
        
        # Dataset Size
        "total_samples": None,
        
        # Performance Optimization
        "apply_amp": enable_amp,
        "use_torch_compile": enable_compile,
        
        # QAT - Disabled for testing
        "apply_qat_representation": False,
        "qat_representation_bits": 8,
        "qat_representation_symmetric": True,
        "qat_representation_start_step": float('inf'),
        
        # Resume Training
        "resume_from_checkpoint": None,
        "resume_learning_rate": 0.002,
        "resume_warmup_steps": 0,
        
        # Wandb - Disabled for testing
        "wandb_project": f"batch-size-test-{batch_size}",
        "wandb_api_key": "",
        "wandb_watch_freq": 1000,
        "disable_wandb_git": True,
    }
    return config

def save_test_config(config, config_path):
    """Save config as Python file"""
    with open(config_path, 'w') as f:
        f.write("# Auto-generated batch size test config\n")
        f.write(f"config = {repr(config)}\n")

def run_batch_test(batch_size, num_gpus=4, max_steps=5, enable_amp=True, enable_compile=False, cache_size=20):
    """Run a batch size test with given parameters"""
    print(f"\n{'='*60}")
    print(f"Testing batch size: {batch_size} (per GPU)")
    print(f"Effective global batch size: {batch_size * num_gpus}")
    print(f"AMP enabled: {enable_amp}, Compile: {enable_compile}, Cache: {cache_size}")
    print(f"{'='*60}")
    
    # Create test config
    config = create_test_config(batch_size, enable_amp, enable_compile, cache_size)
    config_name = f"batch_test_{batch_size}_{int(enable_amp)}_{int(enable_compile)}_{cache_size}.py"
    config_path = CONFIG_DIR / config_name
    
    # Ensure config directory exists
    CONFIG_DIR.mkdir(exist_ok=True)
    save_test_config(config, config_path)
    
    # Command to run training with apptainer (like in sbatch scripts)
    container_image = "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif"
    work_dir = str(SCRIPT_DIR)
    
    # Set up bind paths like in the sbatch script
    parent_dir = str(SCRIPT_DIR.parent)
    bind_paths = f"{work_dir}:{work_dir},{parent_dir}:{parent_dir},/scratch:/scratch"
    
    cmd = [
        "apptainer", "exec", "--nv",
        "--bind", bind_paths,
        "--pwd", work_dir,
        container_image,
        "torchrun",
        "--standalone",
        "--nnodes=1", 
        f"--nproc_per_node={num_gpus}",
        f"{work_dir}/train_ssl_multi_gpu.py",
        "--config", str(config_path)
    ]
    
    start_time = time.time()
    success = False
    max_memory_mb = 0
    peak_memory_per_gpu = []
    
    try:
        # Set environment for testing (based on train_2gpu.sbatch production settings)
        env = os.environ.copy()
        # Keep WANDB enabled for testing as requested
        env["OMP_NUM_THREADS"] = "4"      # Match production settings
        env["MKL_NUM_THREADS"] = "4"
        # NCCL settings for A100s (from train_2gpu.sbatch)
        env["NCCL_SOCKET_IFNAME"] = "ib0,ib1"  # Use InfiniBand interfaces
        env["NCCL_IB_DISABLE"] = "0"            # Enable InfiniBand (A100s have 200Gbps IB)
        env["NCCL_P2P_DISABLE"] = "0"           # Enable P2P (A100s have NVLinks)
        env["NCCL_DEBUG"] = "WARN"              # Less verbose for testing
        # PYTHONPATH for tessera_ml imports
        env["PYTHONPATH"] = f"{parent_dir}:{work_dir}" + (f":{env.get('PYTHONPATH', '')}" if env.get('PYTHONPATH') else "")
        
        # Run the training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=SCRIPT_DIR
        )
        
        step_count = 0
        for line in process.stdout:
            print(line.strip())
            
            # Look for step completion in logs
            if "Step=" in line and "Loss=" in line:
                step_count += 1
                if step_count >= max_steps:
                    print(f"Reached {max_steps} steps, terminating test...")
                    process.terminate()
                    break
            
            # Look for memory information
            if "GPU memory usage" in line and "MB" in line:
                try:
                    memory_str = line.split("GPU memory usage")[1].split("MB")[0]
                    memory_val = float(memory_str.strip().replace(":", "").replace("after FSDP", "").strip())
                    max_memory_mb = max(max_memory_mb, memory_val)
                except:
                    pass
        
        # Wait for process to complete or timeout
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 min timeout
            if process.returncode == 0 or step_count >= max_steps:
                success = True
                print(f"✅ Batch size {batch_size} PASSED")
            else:
                print(f"❌ Batch size {batch_size} FAILED with return code {process.returncode}")
                if stderr:
                    print("STDERR:", stderr[-1000:])  # Last 1000 chars of stderr
        except subprocess.TimeoutExpired:
            print(f"⏰ Batch size {batch_size} TIMEOUT")
            process.kill()
            success = False
        
    except Exception as e:
        print(f"❌ Batch size {batch_size} ERROR: {e}")
        success = False
    
    duration = time.time() - start_time
    
    # Get GPU memory info
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                peak_memory_per_gpu.append({
                    'gpu': i,
                    'allocated_gb': memory_allocated,
                    'reserved_gb': memory_reserved
                })
        torch.cuda.empty_cache()
    except:
        pass
    
    result = {
        'batch_size': batch_size,
        'global_batch_size': batch_size * num_gpus,
        'success': success,
        'duration_sec': duration,
        'max_memory_mb': max_memory_mb,
        'peak_memory_per_gpu': peak_memory_per_gpu,
        'steps_completed': step_count,
        'enable_amp': enable_amp,
        'enable_compile': enable_compile,
        'cache_size': cache_size
    }
    
    # Clean up config file
    try:
        config_path.unlink()
    except:
        pass
    
    return result

def find_optimal_batch_size():
    """Main function to find optimal batch size"""
    print("🚀 Starting batch size optimization for 4 A100 GPUs")
    print(f"Target: Find largest batch size achieving global batch ≥ 4096")
    print(f"System: 4 A100 SXM4 GPUs, 256GB RAM")
    
    results = []
    
    # Test configurations
    test_configs = [
        # Start conservative with proven working config, then scale up
        {"batch_size": 2, "amp": False, "compile": False, "cache": 20},
        {"batch_size": 4, "amp": False, "compile": False, "cache": 20},
        {"batch_size": 8, "amp": True, "compile": False, "cache": 20},
        {"batch_size": 16, "amp": True, "compile": False, "cache": 20},
        {"batch_size": 32, "amp": True, "compile": False, "cache": 20},
        {"batch_size": 64, "amp": True, "compile": False, "cache": 20},
        {"batch_size": 128, "amp": True, "compile": False, "cache": 20},
        {"batch_size": 256, "amp": True, "compile": False, "cache": 20},
        {"batch_size": 512, "amp": True, "compile": False, "cache": 20},
        {"batch_size": 1024, "amp": True, "compile": False, "cache": 20},
        
        # Test with larger cache if smaller batch sizes work
        {"batch_size": 512, "amp": True, "compile": False, "cache": 50},
        {"batch_size": 256, "amp": True, "compile": False, "cache": 100},
        
        # Test with compile if medium batch sizes work
        {"batch_size": 256, "amp": True, "compile": True, "cache": 20},
        {"batch_size": 512, "amp": True, "compile": True, "cache": 20},
    ]
    
    successful_configs = []
    
    for config in test_configs:
        result = run_batch_test(
            batch_size=config["batch_size"],
            enable_amp=config["amp"], 
            enable_compile=config["compile"],
            cache_size=config["cache"]
        )
        results.append(result)
        
        if result['success']:
            successful_configs.append(config)
            print(f"✅ Success: batch_size={config['batch_size']}, global={result['global_batch_size']}")
        else:
            print(f"❌ Failed: batch_size={config['batch_size']}")
            # If we hit OOM, don't test larger batch sizes with same settings
            if result['steps_completed'] == 0:
                print(f"  Skipping larger batch sizes with these settings...")
                # Remove larger batch sizes with same amp/compile settings
                test_configs = [c for c in test_configs 
                              if not (c['batch_size'] > config['batch_size'] 
                                    and c['amp'] == config['amp'] 
                                    and c['compile'] == config['compile'])]
    
    # Find optimal configuration
    print(f"\n{'='*60}")
    print("BATCH SIZE OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    
    if successful_configs:
        # Find config with highest global batch size ≥ 4096
        target_configs = [c for c, r in zip(successful_configs, results) 
                         if r['success'] and r['global_batch_size'] >= 4096]
        
        if target_configs:
            # Get the one with highest batch size
            best_config_idx = max(range(len(target_configs)), 
                                key=lambda i: [r for r in results if r['global_batch_size'] == 
                                             target_configs[i]['batch_size'] * 4][0]['global_batch_size'])
            best_config = target_configs[best_config_idx]
            best_result = [r for r in results if r['batch_size'] == best_config['batch_size'] 
                          and r['enable_amp'] == best_config['amp']][0]
            
            print(f"🎯 OPTIMAL CONFIG FOUND:")
            print(f"   Batch size per GPU: {best_config['batch_size']}")
            print(f"   Global batch size: {best_result['global_batch_size']}")
            print(f"   AMP enabled: {best_config['amp']}")
            print(f"   Compile enabled: {best_config['compile']}")
            print(f"   Cache size: {best_config['cache']}")
            print(f"   Peak memory: {best_result.get('max_memory_mb', 'N/A')} MB")
        else:
            # Find highest successful global batch size
            best_result = max([r for r in results if r['success']], 
                            key=lambda x: x['global_batch_size'])
            best_config = [c for c, r in zip(successful_configs, results) 
                          if r == best_result][0]
            
            print(f"⚠️  TARGET NOT REACHED. Best config found:")
            print(f"   Batch size per GPU: {best_config['batch_size']}")
            print(f"   Global batch size: {best_result['global_batch_size']}")
            print(f"   AMP enabled: {best_config['amp']}")
            print(f"   Compile enabled: {best_config['compile']}")
            print(f"   Cache size: {best_config['cache']}")
            print(f"   Peak memory: {best_result.get('max_memory_mb', 'N/A')} MB")
    else:
        print("❌ No successful configurations found!")
    
    # Print summary table
    print(f"\n{'Batch Size':<10} {'Global':<8} {'AMP':<5} {'Compile':<8} {'Cache':<7} {'Success':<8} {'Memory(MB)':<12}")
    print(f"{'-'*10} {'-'*8} {'-'*5} {'-'*8} {'-'*7} {'-'*8} {'-'*12}")
    for result in results:
        status = "✅" if result['success'] else "❌"
        memory_str = f"{result.get('max_memory_mb', 0):.0f}" if result.get('max_memory_mb') else "N/A"
        print(f"{result['batch_size']:<10} {result['global_batch_size']:<8} {result['enable_amp']!s:<5} "
              f"{result['enable_compile']!s:<8} {result['cache_size']:<7} {status:<8} {memory_str:<12}")
    
    return results, successful_configs

if __name__ == "__main__":
    # Make sure we're in the right directory
    os.chdir(SCRIPT_DIR)
    
    # Run the optimization
    results, successful_configs = find_optimal_batch_size()
    
    # Save results to JSON for later analysis
    results_file = SCRIPT_DIR / "batch_size_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {results_file}")