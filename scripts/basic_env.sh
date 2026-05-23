conda create -n sparsevideo python=3.12.9
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
conda install nvidia::cuda-toolkit==12.8.0
conda install nvidia::cuda-nvcc==12.8.61
conda install nvidia::cuda==12.8.0

pip install setuptools wheel packaging psutil ninja einops

cd kernels
git clone https://github.com/Dao-AILab/flash-attention.git
git submodule update --init csrc/cutlass
export MAX_JOBS=16
export FLASH_ATTN_CUDA_ARCHS="80"
export TORCH_CUDA_ARCH_LIST="8.0"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
python setup.py install
pip install -v --no-build-isolation .
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir -v