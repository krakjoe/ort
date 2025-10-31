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
- Zero configuration simplicity (adjustable via environment)

**Notes:**
- The backend uses a threshold-based system (16KB default) to decide between GPU and CPU execution
- Matrix operations leverage cuBLAS for optimized performance
- All operations include proper stream synchronization to ensure data consistency

## Building with CUDA support

Example given is ubuntu:

```bash
# Install dependencies
sudo apt install nvidia-cuda-toolkit

# Build and install extension
phpize
./configure --enable-ort-backend=cuda,auto --enable-ort
make -j$(nproc) ort-cuda-kernels
make -j$(nproc)
sudo make install
```

## Configuring CUDA

CUDA is configured via environment variables (this means process wide configuration applies):

  - `ORT_CUDA_DEVICE`    - device selector, *integer, pci bus id string (long form)* (default: 0)
  - `ORT_CUDA_THREADS`   - thread count adjuster, *integer* (default: 256)
  - `ORT_CUDA_THRESHOLD` - threshold adjuster, *integer* (default: 16384)

**Note: Use `lspci -D` to find pci bus id of specific device in long form.**

## CUDA Characteristics

Linking against CUDA incurs some overhead, this is observable and unavoidable. While initialization is lazy (ie, CUDA is not fully initialized until an API call that requires it), lazy loading would significantly complicate the implementation.

Where the size of the operand(s) < `ORT_CUDA_THRESHOLD` in bytes, execution is relayed to the CPU backend (which may be SIMD accelerated, or not); Executing on the GPU comes with some overhead to setup the massively parallel execution, for small operations the overhead would exceed the potential gain.

To insulate the programmer from the complexities of managing memory (and also keep the backend as simple as possible) CUDA Unified (Managed) Memory is used to allocate tensors that when used as an operand to a kernel would cause the kernel to execute on the GPU. Tensors of a size below the threshold will be allocated on the CPU.

Unified memory is migrated to the CPU as necessary (upon access); This means subsequent execution of `Math` functions does not incur the overhead of migration (unless an operand is accessed between subsequent executions).