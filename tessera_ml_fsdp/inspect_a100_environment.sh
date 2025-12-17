#!/bin/bash

echo "=== A100 Node Environment Inspection ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo ""

echo "=== GPU Information ==="
nvidia-smi
echo ""

echo "=== Network Interfaces ==="
echo "All network interfaces:"
ip addr show | grep -E "^[0-9]+:|inet "
echo ""

echo "Interface names only:"
ip addr show | grep "^[0-9]" | awk '{print $2}' | sed 's/://'
echo ""

echo "=== Active Routes ==="
ip route show
echo ""

echo "=== Network Interface Details ==="
echo "Available network devices:"
ls -la /sys/class/net/
echo ""

echo "=== InfiniBand Check ==="
if command -v ibstat > /dev/null 2>&1; then
    echo "✅ InfiniBand tools available"
    ibstat 2>/dev/null || echo "❌ ibstat command failed"
    
    # Check for IB devices
    if ls /dev/infiniband/ > /dev/null 2>&1; then
        echo "InfiniBand devices found:"
        ls -la /dev/infiniband/
    else
        echo "No InfiniBand devices found in /dev/infiniband/"
    fi
else
    echo "❌ InfiniBand tools NOT available"
fi
echo ""

echo "=== RDMA Check ==="
if command -v ibv_devinfo > /dev/null 2>&1; then
    echo "RDMA devices:"
    ibv_devinfo || echo "No RDMA devices found"
else
    echo "RDMA tools not available"
fi
echo ""

echo "=== GPU Topology ==="
nvidia-smi topo -m 2>/dev/null || echo "nvidia-smi topology not available"
echo ""

echo "=== NUMA Information ==="
if command -v numactl > /dev/null 2>&1; then
    echo "NUMA topology:"
    numactl --hardware 2>/dev/null || echo "numactl failed"
else
    echo "numactl not available"
fi
echo ""

echo "=== CPU Information ==="
lscpu | grep -E "Socket|Core|Thread|NUMA"
echo ""

echo "=== Memory Information ==="
free -h
echo ""

echo "=== Current Environment Variables ==="
echo "CUDA-related variables:"
printenv | grep -i cuda || echo "No CUDA variables set"
echo ""
echo "NCCL-related variables:"
printenv | grep -i nccl || echo "No NCCL variables set"
echo ""

echo "=== Network Interface Analysis ==="
echo "Physical interfaces (excluding virtual):"
for iface in $(ls /sys/class/net/); do
    if [ -e "/sys/class/net/$iface/device" ]; then
        echo "  $iface (physical)"
    else
        echo "  $iface (virtual)"
    fi
done
echo ""

echo "=== Recommended NCCL Configuration ==="
echo "Based on inspection above:"
echo ""

# Determine which interfaces to exclude
virtual_ifaces=""
for iface in $(ls /sys/class/net/); do
    if [ ! -e "/sys/class/net/$iface/device" ]; then
        if [ -z "$virtual_ifaces" ]; then
            virtual_ifaces="^$iface"
        else
            virtual_ifaces="$virtual_ifaces,^$iface"
        fi
    fi
done

echo "Suggested NCCL_SOCKET_IFNAME: $virtual_ifaces"
echo ""

if command -v ibstat > /dev/null 2>&1 && ibstat > /dev/null 2>&1; then
    echo "InfiniBand detected - can try with NCCL_IB_DISABLE=0"
else
    echo "No InfiniBand - should set NCCL_IB_DISABLE=1"
fi
echo ""

gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "GPU count: $gpu_count"
if [ "$gpu_count" -gt 1 ]; then
    echo "Multiple GPUs - check if P2P communication available"
    echo "nvidia-smi topo -m output above shows GPU connectivity"
else
    echo "Single GPU - P2P not relevant"
fi

echo ""
echo "=== Test Complete ==="