#!/bin/bash

python_executable=python$1
cuda_home=/usr/local/cuda-$2

# Update paths
PATH=${cuda_home}/bin:$PATH
LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

echo "Setting up virtual environment in tmpfs..."
sudo mkdir -p /dev/shm/build_env
$python_executable -m venv /dev/shm/build_env
source /dev/shm/build_env/bin/activate

# Update python executable to use virtual environment
python_executable=/dev/shm/build_env/bin/python

export VIRTUAL_ENV="/dev/shm/build_env"
export PATH="/dev/shm/build_env/bin:$PATH"
export TMPDIR="/dev/shm/tmp"
mkdir -p "$TMPDIR"

# Install requirements in virtual environment
$python_executable -m pip install --upgrade pip wheel setuptools packaging
$python_executable -m pip install -r requirements/cuda.txt

# Limit the number of parallel jobs to avoid OOM and disk space issues
export MAX_JOBS=1
export NVCC_THREADS=1

export TORCH_CUDA_ARCH_LIST="8.0"

(while true; do 
  find /tmp -name "tmpxft_*" -mmin +0.5 -delete 2>/dev/null || true
  find /tmp -name "*cudafe*" -mmin +0.5 -delete 2>/dev/null || true
  find /tmp -name "*.fatbin.c" -mmin +0.5 -delete 2>/dev/null || true
  find /tmp -name "*.stub.c" -mmin +0.5 -delete 2>/dev/null || true
  find /tmp -name "cc*.s" -mmin +0.5 -delete 2>/dev/null || true
  sleep 30
done) &
CLEANUP_PID=$!

# Build
$python_executable setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

kill $CLEANUP_PID 2>/dev/null || true
find /tmp -name "*.fatbin.c" -delete 2>/dev/null || true
find /tmp -name "*cudafe*" -delete 2>/dev/null || true
find /tmp -name "tmpxft_*" -delete 2>/dev/null || true
rm -rf /tmp/.deps 2>/dev/null || true
