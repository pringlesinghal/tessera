#!/usr/bin/env python3
"""
Progressive batch size finder for 4 A100 GPUs
Tests increasing batch sizes until OOM to find the maximum
"""

import os
import subprocess
import time
import json
from pathlib import Path

def create_batch_config(batch_size):
    """Create config for specific batch size"""
    config = f'''# Auto-generated config for batch_size={batch_size}
config = {{
    # Dataset - smaller for faster testing
    "index_dir": "/scratch/groups/dlobell/psinghal/sentineldownloader/tree_index_k750k_shuffled",
    "data_root": "/scratch/users/psinghal/time_series", 
    "years": [2016],  # Single year for speed
    "cache_size": 20,
    
    # TARGET BATCH SIZE
    "batch_size": {batch_size},
    "epochs": 1,
    
    # Training params
    "learning_rate": 0.002,
    "barlow_lambda": 5e-3,
    "warmup_ratio": 0.1,
    "weight_decay": 1e-6,
    "clip_grad_norm": 2.0,
    
    # Model - proven architecture
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
    
    # CRITICAL: Enable AMP for large batches
    "apply_amp": True,
    "use_torch_compile": False,
    "apply_mixup": False,  # Disable for speed
    
    # QAT disabled
    "apply_qat_representation": False,
    "qat_representation_start_step": 999999,
    
    # No resume
    "resume_from_checkpoint": None,
    
    # Wandb
    "wandb_project": "batch-size-search-{batch_size}",
    "wandb_api_key": "",
    "disable_wandb_git": True,
}}'''
    return config

def test_batch_size(batch_size, timeout_sec=300):
    """Test a specific batch size"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING BATCH_SIZE={batch_size} (Global: {batch_size * 4})")
    print(f"{'='*60}")
    
    # Create config file
    config_file = f"test_batch_{batch_size}.py"
    with open(config_file, 'w') as f:
        f.write(create_batch_config(batch_size))
    
    # Environment setup - CRITICAL for fixing socket errors
    env = os.environ.copy()
    env.update({
        "NCCL_SOCKET_IFNAME": "ib0,ib1,em1",
        "NCCL_IB_DISABLE": "0",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_DEBUG": "WARN",
        "NCCL_SOCKET_FAMILY": "AF_INET",  # Force IPv4
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "PYTHONPATH": "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera:/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml"
    })
    
    # Command
    cmd = [
        "apptainer", "exec", "--nv",
        "--bind", "/scratch:/scratch",
        "--pwd", "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml",
        "/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/ml_env.sif",
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=127.0.0.1:29500",
        "train_ssl_multi_gpu.py",
        "--config", config_file
    ]
    
    success = False
    peak_memory_gb = 0
    steps_completed = 0
    error_type = "unknown"
    
    start_time = time.time()
    
    try:
        print(f"Command: {' '.join(cmd[-6:])}")  # Show last part of command
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )
        
        # Monitor output
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                # Print key lines only
                if any(keyword in line for keyword in [
                    "GPU memory usage", "Step=", "Loss=", "ERROR", "OOM", 
                    "out of memory", "CUDA error", "SUCCESS", "Starting training"
                ]):
                    print(f"  {line}")
                
                # Track progress
                if "Step=" in line and "Loss=" in line:
                    steps_completed += 1
                    if steps_completed >= 3:  # Just need a few steps to prove it works
                        print(f"  ✅ REACHED {steps_completed} STEPS - SUCCESS!")
                        process.terminate()
                        success = True
                        break
                
                # Track memory
                if "GPU memory usage" in line and "MB" in line:
                    try:
                        memory_str = line.split("GPU memory usage")[1].split("MB")[0]
                        memory_mb = float(memory_str.replace(":", "").strip())
                        peak_memory_gb = max(peak_memory_gb, memory_mb / 1024)
                    except:
                        pass
                
                # Check for errors
                if any(error in line.lower() for error in ["out of memory", "oom"]):
                    error_type = "OOM"
                    print(f"  ❌ OUT OF MEMORY DETECTED")
                    process.terminate()
                    break
                elif any(error in line.lower() for error in ["cuda error", "runtime error"]):
                    error_type = "CUDA_ERROR"
                    print(f"  ❌ CUDA ERROR DETECTED")
                    process.terminate()
                    break
        
        # Wait for completion
        try:
            process.wait(timeout=max(10, timeout_sec - (time.time() - start_time)))
        except subprocess.TimeoutExpired:
            error_type = "TIMEOUT"
            print(f"  ⏰ TIMEOUT after {timeout_sec}s")
            process.kill()
        
    except Exception as e:
        error_type = "EXCEPTION"
        print(f"  ❌ EXCEPTION: {e}")
    
    # Cleanup
    try:
        os.remove(config_file)
    except:
        pass
    
    duration = time.time() - start_time
    
    result = {
        'batch_size': batch_size,
        'global_batch_size': batch_size * 4,
        'success': success,
        'steps_completed': steps_completed,
        'peak_memory_gb': peak_memory_gb,
        'duration_sec': duration,
        'error_type': error_type
    }
    
    status = "✅ SUCCESS" if success else f"❌ FAILED ({error_type})"
    print(f"  Result: {status}")
    print(f"  Steps: {steps_completed}, Memory: {peak_memory_gb:.1f}GB, Duration: {duration:.1f}s")
    
    return result

def find_maximum_batch_size():
    """Find the maximum working batch size"""
    print("🚀 PROGRESSIVE BATCH SIZE FINDER - 4 A100 GPUs")
    print("Goal: Find maximum batch size without OOM")
    print("Strategy: Binary search starting from proven working sizes")
    
    results = []
    
    # Test sequence - exponential growth then binary search
    test_sequence = [
        # Start from known working
        2,      # Baseline (known working)
        64,     # Small jump 
        256,    # Medium
        512,    # Large
        1024,   # Target (4096 global)
        1536,   # Between 1024-2048
        2048,   # Stretch (8192 global)
        3072,   # Between 2048-4096
        4096,   # Very large (16384 global)
        6144,   # Between 4096-8192
        8192,   # Huge (32768 global)
    ]
    
    max_working_batch = 2  # Start conservatively
    
    for batch_size in test_sequence:
        result = test_batch_size(batch_size, timeout_sec=180)  # 3min timeout
        results.append(result)
        
        if result['success']:
            max_working_batch = batch_size
            print(f"✅ NEW MAX: batch_size={batch_size} (global={batch_size*4})")
        else:
            print(f"❌ FAILED at batch_size={batch_size}")
            if result['error_type'] == 'OOM':
                print("  🛑 OOM detected - stopping search")
                break
    
    # Binary search refinement if we found limits
    if max_working_batch < max(test_sequence) and results and not results[-1]['success']:
        last_failed = results[-1]['batch_size']
        low = max_working_batch
        high = last_failed
        
        print(f"\n🔍 BINARY SEARCH between {low} and {high}")
        
        while high - low > 256:  # Search until gap is small
            mid = (low + high) // 2
            if mid in [r['batch_size'] for r in results]:  # Skip if already tested
                break
                
            result = test_batch_size(mid, timeout_sec=120)
            results.append(result)
            
            if result['success']:
                max_working_batch = mid
                low = mid
                print(f"  ✅ {mid} works, trying higher")
            else:
                high = mid
                print(f"  ❌ {mid} fails, trying lower")
    
    # Final results
    print(f"\n{'='*80}")
    print("BATCH SIZE SEARCH RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Batch/GPU':<10} {'Global':<8} {'Steps':<7} {'Memory':<9} {'Duration':<10} {'Status':<15}")
    print(f"{'-'*10} {'-'*8} {'-'*7} {'-'*9} {'-'*10} {'-'*15}")
    
    for r in results:
        status = "✅ SUCCESS" if r['success'] else f"❌ {r['error_type']}"
        print(f"{r['batch_size']:<10} {r['global_batch_size']:<8} {r['steps_completed']:<7} "
              f"{r['peak_memory_gb']:<9.1f} {r['duration_sec']:<10.1f} {status:<15}")
    
    # Find optimal recommendations
    successful = [r for r in results if r['success']]
    
    if successful:
        max_result = max(successful, key=lambda x: x['batch_size'])
        target_4096 = [r for r in successful if r['global_batch_size'] >= 4096]
        
        print(f"\n🎯 OPTIMAL RECOMMENDATIONS:")
        print(f"   Maximum working: batch_size={max_result['batch_size']} (global={max_result['global_batch_size']})")
        print(f"   Peak memory: {max_result['peak_memory_gb']:.1f} GB per GPU")
        
        if target_4096:
            best_4096 = max(target_4096, key=lambda x: x['batch_size'])
            print(f"   ✅ Target achieved: batch_size={best_4096['batch_size']} (global={best_4096['global_batch_size']})")
        else:
            print(f"   ⚠️ Target 4096 not reached. Max global batch: {max_result['global_batch_size']}")
        
        # Memory utilization
        total_memory_gb = max_result['peak_memory_gb']
        utilization = (total_memory_gb / 80) * 100  # A100 has 80GB
        print(f"   Memory utilization: {utilization:.1f}% of A100 capacity")
        
    else:
        print(f"\n❌ NO SUCCESSFUL CONFIGURATIONS!")
        print("   Check NCCL settings and model compatibility")
    
    # Save results
    with open('batch_size_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to batch_size_search_results.json")
    return results

if __name__ == "__main__":
    os.chdir("/scratch/groups/dlobell/psinghal/sentineldownloader/tessera/tessera_ml")
    results = find_maximum_batch_size()