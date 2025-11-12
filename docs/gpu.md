# GPU

## Differences from CPU Backends

CPU backends are required to declare an installation routine, which patches the dispatch table with SIMD accelerated versions of math kernels. The frontend manages the distribution of the computation across the CPU.

In the case of GPU devices, this would not be desirable - GPU devices have their own threading model, and distribution across the CPU would introduce unecessary synchronization of the device and intefere with their threading model; Instead the installation function for GPU backends should make a copy of the dispatch table, installing GPU kernels in the copy, and tagging the original dispatch table (see cuda code for details).

GPU devices are required to implement:

```
void* ort_math_backend_gpu_kernel(
    void* kernel, ONNXTensorElementDataType type,
    ort_tensor_t* result, size_t argc, ...);
```

This function is invoked by the frontend during scheduling; If a kernel is returned, it is immediately entered and scheduling on the CPU is skipped.

## Currently implemented GPU backends:

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

### Building with CUDA support

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

### Configuring CUDA

CUDA is configured via environment variables (this means process wide configuration applies):

  - `ORT_CUDA_DEVICE`    - device selector, *integer, pci bus id string (long form)* (default: 0)
  - `ORT_CUDA_THREADS`   - thread (per block) count adjuster, *integer* (default: 256)
  - `ORT_CUDA_THRESHOLD` - threshold adjuster, *integer*, bytes (default: 16384)

**Note: Use `lspci -D` to find pci bus id of specific device in long form.**

### CUDA Characteristics

Linking against CUDA incurs some overhead, this is observable and unavoidable. While initialization is lazy (ie, CUDA is not fully initialized until an API call that requires it), lazy loading would significantly complicate the implementation.

To insulate the programmer from the complexities of managing memory (and also keep the backend as simple as possible) CUDA Unified (Managed) Memory is used to allocate tensors above `ORT_CUDA_THRESHOLD`.

The `ort_math_backend_gpu_kernel` implementation will determine if the math kernel has a CUDA implementation, it will then determine if the operand(s) are allocated in CUDA managed memory (possibly migrating the result to CUDA managed memory), if all preconditions are met the kernel is returned (and so frontend scheduling is skipped).

Unified memory is migrated to the CPU as necessary (upon access); This means subsequent execution of `Math` functions does not incur the overhead of migration (unless an operand is accessed between subsequent executions).