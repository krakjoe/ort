# GPU

Currently implemented GPU backends:

 - CUDA (nvidia)

### CUDA Requirements

The CUDA backend requires:

**Hardware:**
- NVIDIA GPU with compute capability **sm_60** or higher (Pascal architecture and newer)
- Compatible GPUs include:
  - GeForce GTX 10xx series and newer
  - Tesla P-series and newer
  - Quadro P-series and newer
  - RTX series (all generations)

**Software:**
- CUDA Toolkit **11.0 or higher** (developed and tested with CUDA 12.0)
- NVIDIA driver version compatible with your CUDA toolkit
- C++14 compatible compiler (required for CUDA kernel compilation)

**Build Dependencies:**
- `nvcc` CUDA compiler
- `libcudart` (CUDA Runtime Library)
- `libcublas` (CUDA Basic Linear Algebra Subroutines)

**Note: all of these dependencies are part of the nvidia-cuda-toolkit (on ubuntu)**

**Runtime Features:**
- Uses CUDA Managed Memory for seamless CPU/GPU memory management
- Automatic fallback to CPU backend when GPU operations fail
- Environment variable `ORT_CUDA_DEVICE` for device selection
- Environment variable `ORT_CUDA_THRESHOLD` to adjust threshold

**Notes:**
- The backend uses a threshold-based system (16KB default) to decide between GPU and CPU execution
- Matrix operations leverage cuBLAS for optimized performance
- All operations include proper stream synchronization to ensure data consistency

## Building with CUDA support

The CPU backend is automatically selected by the build process, however it's not practical so automatically build with CUDA support based on detected presence in the same way as CPU backend selection.

On most modern platforms (of the kind where you might also have a nvidia card, example given is ubuntu):

```bash
# Install dependencies
sudo apt install nvidia-cuda-toolkit

# Build and install extension
phpize
./configure --enable-ort-backend=cuda,avx2 --enable-ort
make -j$(nproc) ort-cuda-kernels
make -j$(nproc)
sudo make install
```