# PHP-ORT

<em>PHP - Oh Really, Tensors?</em>

This extension provides high performance tensor mathematics. It supports a wide range of mathematical operations on tensors, including element-wise computations, reductions, linear algebra functions, and more.

This extension also includes optional integration with Microsoft's ONNX Runtime for efficient model execution.

## Features

- **High-Performance Mathematics**: SIMD-accelerated operations (WASM, NEON, RISCV64, AVX2, SSE4.1, SSE2)
- **Multi-Core Parallelism**: Automatic work distribution across CPU cores
- **Comprehensive Type Support**: 9 data types with automatic promotion
- **Memory Efficient**: Reference-counted tensors with zero-copy slicing
- **ONNX Integration**: Optional and Direct ONNX model loading and inference

## Quick Start

### Installation

```bash
# Build extension
phpize
./configure --enable-ort
make
sudo make install
```

### Installation with ONNX Runtime

```bash
# Fetch onnxruntime from github
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz

# Install onnxruntime
tar -C /usr/local -xvf onnxruntime-linux-x64-1.22.0.tgz --strip-components 1

# Build and install extension
phpize
./configure --with-ort-onnx=/usr/local --enable-ort
make
sudo make install
```

<em>Note: Use microsoft provided releases of ONNX, not brew or ppa or apt provided releases.</em>

#### Post Installation

Add to your `php.ini`:
```ini
extension=ort.so
```

### Basic Usage

```php
use ORT\Tensor\Transient;
use ORT\Math;

// Create tensors
$a = new Transient([1000, 1000], $matrix_data, ORT\Tensor::FLOAT);
$b = new Transient([1000, 1000], $matrix_data, ORT\Tensor::FLOAT);

// High-performance matrix multiplication
$result = Math\matmul($a, $b);

// Element-wise operations
$sum = Math\add($a, $b);
$scaled = Math\multiply($a, 2.5);  // Scalar operations

// Reductions
$total = Math\reduce\tensor\sum($a);
$row_sums = Math\reduce\axis\sum($a, 1);
```

## Performance

- **SIMD Acceleration**: Up to 8x faster with AVX2 vectorization
- **Multi-Core Scaling**: Automatic scaling to available CPU cores
- **Memory Optimization**: Aligned memory allocation for optimal performance

Check your system configuration:
```php
echo "Backend: " . (ORT\Math\backend() ?: "scalar") . "\n";
echo "Cores: " . ORT\Math\cores() . "\n";
```

## API Reference (not exhaustive)

### Tensor Types
- `ORT\Tensor\Transient` - Temporary tensors
- `ORT\Tensor\Persistent` - Named, persistent tensors

### Mathematical Operations
- **Binary**: `add`, `subtract`, `multiply`, `divide`, `pow`, `mod`
- **Unary**: `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `neg`
- **Matrix**: `matmul`, `dot`, `transpose`
- **Reduction**: `sum`, `mean`, `min`, `max`, `softmax`

### ONNX Integration
- `ORT\Model` - Load and manage ONNX models
- `ORT\Runtime` - Execute ONNX model inference

## Configuration

### Environment Variables
- `ORT_SCALE_CORES` - Set thread pool size (default: CPU cores)

### Build Options
- `--enable-ort-backend=type`  - Enable SIMD optimizations (default: auto)
 - `auto`    - detect backend type for platform
 - `wasm`    - use wasm backend
 - `neon`    - use neon backend
 - `riscv64` - use riscv64 backend
 - `avx2`    - use avx2 backend
 - `sse41`   - use sse41 backend
 - `sse2`    - use sse2 backend
 - `none`    - use no backend
- `--with-ort-onnx`       - Link against ONNX Runtime (default: no)

## Technical Details

See [docs](docs) for detailed technical documentation.

## Requirements

- **PHP**: 7.4+ or 8.0+
- **ONNX Runtime**: 1.16+ (optional)
- **Compiler**: GCC 4.8+ or Clang 3.8+
- **CPU**: x86_64 with SSE2 (AVX2 recommended)

## License

This project is licensed under the PHP License 3.01.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- SIMD implementations include proper fallbacks
- Thread safety is maintained
- Tests pass across supported PHP versions

## Support

For issues and feature requests, please use the project's issue tracker.