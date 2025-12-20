#!/bin/bash
# CI Cleanup Script for Deep Tree Echo AGI
# Aggressively frees disk space before builds

set -e

echo "============================================================"
echo "CI AGGRESSIVE CLEANUP FOR DEEP TREE ECHO AGI"
echo "============================================================"

# Record initial space
echo ""
echo "=== Initial Disk Space ==="
df -h /
echo ""

# Calculate initial available space
initial_space=$(df / | tail -1 | awk '{print $4}')
echo "Initial available space: ${initial_space}"

# Remove large unused software
echo ""
echo "=== Removing Unused Software ==="

if [ -d "/usr/share/dotnet" ]; then
    echo "Removing .NET..."
    sudo rm -rf /usr/share/dotnet
    echo "✅ .NET removed"
fi

if [ -d "/opt/ghc" ]; then
    echo "Removing Haskell (GHC)..."
    sudo rm -rf /opt/ghc
    echo "✅ Haskell removed"
fi

if [ -d "/usr/local/share/boost" ]; then
    echo "Removing Boost libraries..."
    sudo rm -rf /usr/local/share/boost
    echo "✅ Boost removed"
fi

if [ -n "$AGENT_TOOLSDIRECTORY" ] && [ -d "$AGENT_TOOLSDIRECTORY" ]; then
    echo "Removing Agent Tools Directory..."
    sudo rm -rf "$AGENT_TOOLSDIRECTORY"
    echo "✅ Agent Tools removed"
fi

if [ -d "/usr/local/lib/android" ]; then
    echo "Removing Android SDK..."
    sudo rm -rf /usr/local/lib/android
    echo "✅ Android SDK removed"
fi

if [ -d "/usr/local/lib/node_modules" ]; then
    echo "Removing large Node modules..."
    sudo rm -rf /usr/local/lib/node_modules
    echo "✅ Node modules removed"
fi

if [ -d "/opt/hostedtoolcache" ]; then
    echo "Removing hosted toolcache..."
    sudo rm -rf /opt/hostedtoolcache
    echo "✅ Hosted toolcache removed"
fi

# Clean package caches
echo ""
echo "=== Cleaning Package Caches ==="

echo "Cleaning APT cache..."
sudo apt-get clean || true
sudo apt-get autoclean || true
sudo apt-get autoremove -y || true
echo "✅ APT cache cleaned"

echo "Cleaning pip cache..."
pip cache purge 2>/dev/null || true
echo "✅ Pip cache cleaned"

# Clean Docker (if available)
if command -v docker >/dev/null 2>&1; then
    echo ""
    echo "=== Cleaning Docker ==="
    docker system prune -af 2>/dev/null || true
    echo "✅ Docker cleaned"
fi

# Clean temporary files
echo ""
echo "=== Cleaning Temporary Files ==="

echo "Cleaning /tmp..."
sudo find /tmp -type f -atime +0 -delete 2>/dev/null || true
echo "✅ /tmp cleaned"

echo "Cleaning CUDA artifacts..."
find /tmp -name "*.fatbin.c" -delete 2>/dev/null || true
find /tmp -name "*.cudafe*" -delete 2>/dev/null || true
find /tmp -name "tmpxft_*" -delete 2>/dev/null || true
echo "✅ CUDA artifacts cleaned"

# Clean user caches
echo ""
echo "=== Cleaning User Caches ==="

for cache_dir in /root/.cache /home/*/.cache; do
    if [ -d "$cache_dir" ]; then
        echo "Cleaning $cache_dir..."
        sudo rm -rf "$cache_dir" 2>/dev/null || true
    fi
done
echo "✅ User caches cleaned"

# Record final space
echo ""
echo "=== Final Disk Space ==="
df -h /
echo ""

# Calculate freed space
final_space=$(df / | tail -1 | awk '{print $4}')
echo "Final available space: ${final_space}"

# Calculate difference (approximate)
echo ""
echo "============================================================"
echo "CLEANUP COMPLETE"
echo "============================================================"
echo "✅ Disk space optimized for AGI build"
echo ""

exit 0
