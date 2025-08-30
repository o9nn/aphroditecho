#!/bin/bash

python_version=${1:-3.11}
cuda_version=${2:-12.4}
cuda_home=/usr/local/cuda-${cuda_version}

if command -v python${python_version} >/dev/null 2>&1; then
    python_executable=python${python_version}
elif command -v python3 >/dev/null 2>&1; then
    python_executable=python3
elif command -v python >/dev/null 2>&1; then
    python_executable=python
elif [ -n "$PYENV_ROOT" ] && command -v pyenv >/dev/null 2>&1; then
    pyenv global ${python_version}.* 2>/dev/null || pyenv global 3.11.* 2>/dev/null || true
    if command -v python3 >/dev/null 2>&1; then
        python_executable=python3
    elif command -v python >/dev/null 2>&1; then
        python_executable=python
    else
        echo "Error: No Python executable found even with pyenv"
        exit 1
    fi
else
    echo "Error: No Python executable found"
    exit 1
fi

echo "Using Python executable: $python_executable"
$python_executable --version

# Update paths
PATH=${cuda_home}/bin:$PATH
LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

echo "Checking for existing virtual environment..."
if [ -d "/dev/shm/build_env" ] && [ -f "/dev/shm/build_env/bin/python" ]; then
    echo "Using existing virtual environment at /dev/shm/build_env"
    source /dev/shm/build_env/bin/activate
    python_executable=/dev/shm/build_env/bin/python
else
    echo "Creating new virtual environment in tmpfs..."
    sudo mkdir -p /dev/shm/build_env
    sudo chown -R $(whoami):$(whoami) /dev/shm/build_env
    $python_executable -m venv /dev/shm/build_env
    source /dev/shm/build_env/bin/activate
    python_executable=/dev/shm/build_env/bin/python
    
    # Install requirements in virtual environment
    $python_executable -m pip install --upgrade pip wheel setuptools packaging
    $python_executable -m pip install -r requirements/cuda.txt
fi

export VIRTUAL_ENV="/dev/shm/build_env"
export PATH="/dev/shm/build_env/bin:$PATH"
export TMPDIR="/dev/shm/tmp"
mkdir -p "$TMPDIR"

# Limit the number of parallel jobs to avoid OOM and disk space issues
export MAX_JOBS=${MAX_JOBS:-1}
export NVCC_THREADS=${NVCC_THREADS:-1}

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
echo "Starting wheel build process..."
echo "Current working directory: $(pwd)"
echo "Python executable: $python_executable"

PROJECT_ROOT=$(pwd)
mkdir -p "$PROJECT_ROOT/dist"

$python_executable setup.py bdist_wheel --dist-dir="$PROJECT_ROOT/dist" --py-limited-api=cp38

echo "Build completed. Checking for wheel files..."
echo "Python executable used: $python_executable"
echo "Project root: $PROJECT_ROOT"
echo "Virtual environment: $VIRTUAL_ENV"
ls -la "$PROJECT_ROOT/dist/" || echo "No dist directory found"
find "$PROJECT_ROOT" -name "*.whl" -type f || echo "No wheel files found anywhere"
echo "Current working directory contents:"
ls -la "$PROJECT_ROOT"

kill $CLEANUP_PID 2>/dev/null || true
find /tmp -name "*.fatbin.c" -delete 2>/dev/null || true
find /tmp -name "*cudafe*" -delete 2>/dev/null || true
find /tmp -name "tmpxft_*" -delete 2>/dev/null || true
rm -rf /tmp/.deps 2>/dev/null || true
