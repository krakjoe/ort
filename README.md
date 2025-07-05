
# PHP ORT - ONNX Runtime PHP Extension

A high-performance PHP extension providing native bindings to [ONNX Runtime](https://onnxruntime.ai/), enabling efficient machine learning inference and advanced tensor math in PHP applications.

---

## Features
- **Native ONNX Runtime Integration**: Run ONNX models directly from PHP with minimal overhead.
- **SIMD-Accelerated Math**: Optimized math kernels for AVX2, SSE4.1, and SSE2, with automatic scalar fallback for unsupported types or hardware.
- **Comprehensive Tensor Support**: Multi-dimensional arrays, type-safe operations, and memory-efficient reference counting.
- **Session & Model Management**: Fast model loading, session reuse, and flexible runtime configuration.
- **Strict Test & Coverage Workflow**: Enforced build, test, and coverage standards for reliability.

---

## Requirements
- PHP 7.4 or higher
- ONNX Runtime C library (see [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases))
- GCC/Clang, make, autotools

---

## Installation

```bash
# Install ONNX Runtime (see upstream docs)
# Clone and build the extension
phpize
./configure --with-ort --enable-ort-coverage
make
make install
# Add to php.ini:

```

---

## Quick Start

```php
<?php
$model = new ONNX\Model("my_model", "/path/to/model.onnx");
$input = new ONNX\Tensor("input", [1, 3, 224, 224], $data, ONNX\Tensor::FLOAT);
$options = new ONNX\Options();
$runtime = new ONNX\Runtime($model, $options);
$output = $runtime->run(["input" => $input]);
var_dump($output["output"]->getData());
?>
```

---

## API Overview

### ONNX\Model
- Load and inspect ONNX models.

### ONNX\Tensor
- Multi-dimensional, type-safe arrays for model input/output and math operations.
- Supported types: FLOAT, DOUBLE, INT32, INT64, (UINT64 unsupported).

### ONNX\Math
- Elementwise math: `add`, `multiply`, `sqrt`, `abs`, `negate`, `sub`, `div`, `ceil`, `floor`, `round`, `sign`, `reciprocal` (see code for full list).
- All operations fallback to scalar for unsupported types/hardware.

### ONNX\Options
- Configure execution providers, optimization, and session options.

### ONNX\Runtime
- Manage inference sessions and run models.

---

## SIMD & Math Backend
- SIMD kernels for AVX2, SSE4.1, SSE2 (x86/x64 only).
- Scalar fallback for unsupported types (e.g., UINT64, 32-bit int multiply on SSE2).
- All kernels are contract-compliant and tested for correctness and coverage.
- No runtime SIMD dispatch (compile-time selection only).

---

## Development Workflow

1. **Build**: `make clean install` after any source change.
2. **Test**: Always run `make ort-test-coverage-html` to ensure tests run in the correct environment and coverage is generated in `./coverage.info` and `./html/`.
3. **Coverage**: Refer to `./coverage.info` for math layer coverage, not just static analysis.
4. **Optimization**: Follow and update planning documents (see `*_WORKFLOW.md`, `*_PLAN.md`).

---

## Project Structure

- `src/`         — Core C sources (math, tensor, model, runtime, options, status, iterator)
- `src/maths/simd/` — SIMD math backends (AVX2, SSE4.1, SSE2)
- `tests/`       — PHPT and fixture tests
- `html/`        — Coverage reports
- `examples/`    — Example PHP scripts
- `*.md`         — Planning, workflow, and optimization documents

---

## Limitations & Unsupported Features
- **UINT64 tensors** are not supported for most math operations.
- No runtime SIMD dispatch (best SIMD backend is selected at build time).
- Some ONNX types and advanced math ops (e.g., reductions, matrix ops) may be unimplemented.
- No ARM NEON or AVX-512 support (x86/x64 only).

---

## Contributing
- Fork, branch, and submit PRs with tests.
- Follow the enforced build/test/coverage workflow.
- Update planning documents for any optimization or architectural changes.

---

## License
- PHP License v3.01 (see LICENSE)
- Copyright (c) Joe Watkins 2025

---

## Credits
- Author: Joe Watkins (krakjoe)
- ONNX Runtime: Microsoft and contributors

---

## Support & Documentation
- See `tests/` for usage examples
- See planning and workflow docs for development process
- Open issues with detailed reproduction steps

---

*PHP ORT brings high-performance machine learning and tensor math to PHP, with a focus on reliability, maintainability, and production readiness.*
```
