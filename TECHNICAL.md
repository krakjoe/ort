# PHP-ORT: Technical Architecture & Implementation

## Overview

PHP-ORT is a PHP extension that provides high-performance tensor mathematics and machine learning inference capabilities through integration with ONNX Runtime. The extension is architected in multiple layers providing both standalone mathematical operations and ONNX model inference capabilities.

## Architecture

### Core Components

#### 1. Extension Core (`src/ort.c`)
- **ONNX Runtime Integration**: Direct binding to ONNX Runtime C API (`OrtGetApiBase()->GetApi(ORT_API_VERSION)`)
- **Module Lifecycle**: Manages initialization/shutdown of all subsystems
- **Thread Safety**: Uses atomic operations for reference counting with platform-specific implementations (Windows/POSIX)

#### 2. Tensor System (`src/tensor.c`)
- **Dual Storage Model**: 
  - **Persistent Tensors**: Named, heap-allocated tensors with global hash table storage
  - **Transient Tensors**: Temporary, request-scoped tensors with Zend memory management
- **Type System**: Native support for 9 data types (float, double, int8/16/32/64, uint8/16/32, bool)
- **Memory Management**: Reference-counted with atomic operations for thread safety
- **Data Validation**: Recursive shape/data validation with comprehensive error handling
- **Advanced Operations**: Built-in transpose, slicing, and data extraction capabilities

#### 3. Mathematics Layer (`src/maths/`)

##### Dispatch System (`dispatch.c`)
- **Type-Specific Dispatch**: Static dispatch table with 439 function pointers across 9 data types
- **Operation Coverage**: 25+ mathematical operations per supported type
- **Null Handling**: Graceful degradation for unsupported type/operation combinations

##### Frontend Layer (`frontend/`)
- **Unified Interface**: Consistent API across all mathematical operations
- **Broadcasting Support**: Automatic tensor broadcasting for binary operations
- **Scalar Operations**: Dedicated scalar-tensor operation implementations

##### Backend Layer (`backend/`)
- **SIMD Optimization**: Runtime-detected SIMD backend selection (AVX2 > SSE4.1 > SSE2)
- **Vectorized Operations**: 
  - **AVX2**: 256-bit vectors (8 floats, 4 doubles, 32 int8s, 16 int16s, 8 int32s, 4 int64s)
  - **SSE4.1**: 128-bit vectors with enhanced instruction set
  - **SSE2**: 128-bit vectors with basic instruction set
- **Fallback Handling**: Automatic fallback to scalar operations for remainder elements

##### Thread Pool (`pool.c`)
- **Work Distribution**: Cross-platform thread pool implementation (Windows/POSIX)
- **Task Types**: Specialized workers for binary, unary, scalar, reduction, matrix multiplication, and casting operations
- **Dynamic Scaling**: Runtime thread count adjustment via `ORT_POOL_CORES` environment variable
- **Load Balancing**: Chunk-based work distribution with atomic counters

#### 4. PHP API Layer (`src/maths.c`)
- **Namespace Organization**: Functions exposed under `ONNX\Math\*` namespace
- **Macro-Generated Bindings**: Consistent parameter validation and error handling
- **Type Promotion**: Automatic type resolution for mixed-type operations
- **Schema System**: Runtime type compatibility checking

### Mathematical Operations

#### Binary Operations
- **Element-wise**: add, subtract, multiply, divide, mod, pow
- **Broadcasting**: Automatic shape compatibility resolution
- **Scalar Support**: Optimized tensor-scalar operations

#### Unary Operations
- **Trigonometric**: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
- **Exponential**: exp, exp2, log, log2, log10, sqrt, cbrt
- **Rounding**: ceil, floor, round, trunc
- **Utility**: abs, neg, sign, recip

#### Matrix Operations
- **Matrix Multiplication**: Optimized SIMD implementations with proper accumulation
- **Dot Product**: Vector dot product implementation
- **Transpose**: In-place and out-of-place tensor transposition

#### Reduction Operations
- **Tensor Reduction**: Full tensor reduction to scalar (sum, mean, min, max)
- **Axis Reduction**: Reduction along specified axes with keepdims support
- **Softmax**: Numerically stable softmax implementation

### Performance Characteristics

#### SIMD Acceleration
Based on actual implementation analysis:

**AVX2 Performance (256-bit vectors):**
- Float operations: 8 elements per instruction
- Double operations: 4 elements per instruction  
- Integer operations: 8-32 elements per instruction depending on width
- Matrix multiplication: Uses vectorized inner loops with proper accumulation

**Memory Alignment:**
- Custom memory allocator (`ort_alloc`) ensures proper alignment for SIMD operations
- Fallback to unaligned loads (`_mm256_loadu_ps`) when needed

**Thread Pool Utilization:**
- Default thread count: System core count (detectable via `sysconf(_SC_NPROCESSORS_ONLN)`)
- Work distribution: Chunk-based splitting with atomic work stealing
- Task synchronization: Condition variables with proper cleanup

#### Memory Management
- **Reference Counting**: Atomic operations for thread-safe reference management
- **Memory Pools**: Separate allocation strategies for persistent vs transient tensors
- **Slice Views**: Zero-copy tensor slicing with parent reference tracking

### ONNX Integration

#### Model Operations
- **Model Loading**: Direct ONNX model loading via ONNX Runtime C API
- **Runtime Sessions**: Persistent runtime sessions with thread-local storage
- **Memory Management**: Proper cleanup of ONNX Runtime resources

#### Type Conversion
- **Automatic Casting**: Built-in type conversion between PHP and ONNX types
- **Data Marshaling**: Efficient data transfer between PHP arrays and ONNX tensors
- **Error Handling**: Comprehensive error propagation from ONNX Runtime

### Build System

#### Backend Selection
From `config.m4` analysis:
- **Automatic Detection**: Runtime detection of best available instruction set
- **Manual Override**: Configure flags for specific instruction sets
- **Fallback Strategy**: Graceful degradation when SIMD not available

#### Dependencies
- **ONNX Runtime**: Required dependency via pkg-config (`libonnxruntime >= 1.16`)
- **Platform Support**: Windows and POSIX systems
- **Compiler Features**: Atomic operations detection and fallback

### Thread Safety

#### Atomic Operations
Platform-specific atomic implementations:
- **Windows**: `InterlockedAdd`, `InterlockedIncrement`
- **Modern GCC**: `__atomic_add_fetch`, `__atomic_sub_fetch`
- **Legacy GCC**: `__sync_add_and_fetch`, `__sync_sub_and_fetch`

#### Synchronization
- **Persistent Tensors**: Global mutex protection for hash table operations
- **Thread Pool**: Condition variable-based work distribution
- **Reference Counting**: Lock-free atomic reference counting

### Error Handling

#### Exception System
- **Typed Exceptions**: Specific exception classes for different error types
- **Error Propagation**: Proper error handling from ONNX Runtime
- **Validation**: Comprehensive input validation with descriptive error messages

#### Status Flow
Custom `php_ort_status_flow` macro provides:
- **Automatic Cleanup**: Resource cleanup on error conditions
- **Exception Throwing**: Consistent exception handling
- **Error Messages**: Detailed error information with context

### Installation & Configuration

#### Build Process
```bash
# Configure with automatic backend detection
./configure --enable-ort

# Manual backend selection
./configure --enable-ort --disable-ort-avx2  # Force SSE4.1
./configure --enable-ort --disable-ort-backend  # Disable all SIMD
```

#### Runtime Configuration
```bash
# Set thread pool size
export ORT_POOL_CORES=8

# Check runtime configuration
php -r "var_dump(ONNX\\Math\\cores(), ONNX\\Math\\backend());"
```

### API Examples

#### Basic Operations
```php
use ONNX\Tensor\Transient;
use ONNX\Math;

$a = new Transient([1000, 1000], $data, ONNX\Tensor::FLOAT);
$b = new Transient([1000, 1000], $data, ONNX\Tensor::FLOAT);

// Vectorized addition across all cores
$result = Math\add($a, $b);

// Matrix multiplication with SIMD acceleration
$result = Math\matmul($a, $b);
```

#### Advanced Usage
```php
// Thread scaling
$result = Math\scale(4, function() use ($a, $b) {
    return Math\matmul($a, $b);  // Use only 4 cores
});

// Type promotion
$schema = new Math\Schema("add");
$result_type = $schema->resolve(ONNX\Tensor::FLOAT, ONNX\Tensor::DOUBLE);
```

This implementation provides a production-ready, high-performance tensor mathematics library with comprehensive ONNX integration, demonstrating sophisticated engineering in memory management, SIMD optimization, and thread pool design.