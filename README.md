# PHP-ORT

A high-performance PHP extension providing tensor mathematics and ONNX Runtime integration with SIMD acceleration and multi-core parallelism.

## Features

- **High-Performance Mathematics**: SIMD-accelerated operations (AVX2, SSE4.1, SSE2)
- **Multi-Core Parallelism**: Automatic work distribution across CPU cores
- **ONNX Integration**: Direct ONNX model loading and inference
- **Comprehensive Type Support**: 9 data types with automatic promotion
- **Memory Efficient**: Reference-counted tensors with zero-copy slicing

## Quick Start

### Installation

```bash
# Install dependencies
sudo apt-get install libonnxruntime-dev pkg-config

# Build extension
phpize
./configure --enable-ort
make
sudo make install
```

Add to your `php.ini`:
```ini
extension=ort.so
```

### Basic Usage

```php
use ONNX\Tensor\Transient;
use ONNX\Math;

// Create tensors
$a = new Transient([1000, 1000], $matrix_data, ONNX\Tensor::FLOAT);
$b = new Transient([1000, 1000], $matrix_data, ONNX\Tensor::FLOAT);

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
echo "Backend: " . (ONNX\Math\backend() ?: "scalar") . "\n";
echo "Cores: " . ONNX\Math\cores() . "\n";
```

## API Reference

### Tensor Types
- `ONNX\Tensor\Transient` - Temporary tensors
- `ONNX\Tensor\Persistent` - Named, persistent tensors

### Mathematical Operations
- **Binary**: `add`, `subtract`, `multiply`, `divide`, `pow`, `mod`
- **Unary**: `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `neg`
- **Matrix**: `matmul`, `dot`, `transpose`
- **Reduction**: `sum`, `mean`, `min`, `max`, `softmax`

### ONNX Integration
- `ONNX\Model` - Load and manage ONNX models
- `ONNX\Runtime` - Execute model inference
- `ONNX\Math\cast` - Type conversion between tensor types

## Configuration

### Environment Variables
- `ORT_POOL_CORES` - Set thread pool size (default: CPU cores)

### Build Options
- `--enable-ort-backend` - Enable SIMD optimizations (default: yes)
- `--disable-ort-avx2` - Disable AVX2, use SSE4.1
- `--disable-ort-sse41` - Disable SSE4.1, use SSE2
- `--disable-ort-backend` - Disable all SIMD optimizations

## Technical Details

See [TECHNICAL.md](TECHNICAL.md) for detailed architecture documentation, including:
- SIMD implementation details
- Thread pool architecture
- Memory management strategies
- Performance characteristics

## Requirements

- **PHP**: 7.4+ or 8.0+
- **ONNX Runtime**: 1.16+
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