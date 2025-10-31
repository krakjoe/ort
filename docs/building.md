# Building the ORT PHP Extension

This document explains how to build the PHP-ORT extension with various configuration options.

### Core Options

| Option                         | Description                        | Default  |
|--------------------------------|------------------------------------|----------|
| `--enable-ort`                 | Enable the ORT extension           | disabled |
| `--enable-ort-backend[=TYPE]`  | Backend optimization type          | `auto`   |
| `--enable-ort-pool`            | Enable thread pooling              | `yes`    |
| `--with-ort-onnx`              | Enable ONNX Runtime support        | `no`     |

### Backend Types

- **`auto`**: Automatically select the best (as applicable) CPU backend
- **`none`**: Disable backend optimizations
- **`wasm`**: WebAssembly SIMD backend (CPU, *nix only)
- **`neon`**: ARM NEON backend (CPU, *nix only)
- **`riscv64`**: RISCV64 backend (CPU, *nix only)
- **`avx2`**: AVX2 backend (CPU)
- **`sse41`**: SSE4.1 backend (CPU)
- **`sse2`**: SSE2 backend (CPU)
- **`cuda`**: CUDA backend (GPU, *nix only)

## Backend Priority Order

When using `--enable-ort-backend=auto`, the CPU backend is selected in this priority order:

1. **WASM** (highest priority)
2. **NEON**
3. **RISCV64**
3. **AVX2**
4. **SSE4.1**
5. **SSE2** (lowest priority)

## Special Considerations

### WASM Backend Notes

- Forces `ext_shared=no` (static linking only)
- Disables thread pooling (`PHP_ORT_POOL=no`)
- Requires in-tree builds (source directory must equal build directory)
- See [https://github.com/krakjoe/em](krakjoe/em) for building webasm runtime

### CUDA Backend

*For comprehensive information see [gpu.md](gpu.md).*

To build with cuda, combine with appropriate CPU backend (or use automatic selection):

```
--enable-ort-backend=cuda,auto
```

Only one GPU and one CPU backend may be selected.

**Note: Disables `auto`, this means manual selection of the applicable CPU backend is required.**

### ONNX Runtime Integration

When `--with-ort-onnx` is enabled:
 - Requires `libonnxruntime >= 1.16`
 - Uses pkg-config for library detection
 - Adds additional source files for runtime support

Currently it's neceessary to fetch `libonnxruntime` releases from github:

```shell
$ wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz -O onnxruntime-linux-x64-1.22.0.tgz
$ sudo tar -C /usr/local --strip-components=1 -xvzf /onnxruntime-linux-x64-1.22.0.tgz
$ sudo ldconfig # update dynamic linker cache
```

## Build Examples

### Optimized Build with Auto Backend
```bash
./configure --enable-ort --enable-ort-backend=auto
make
```

### Specific Backend Build
```bash
./configure --enable-ort --enable-ort-backend=avx2
make
```

### Full-Featured Build
```bash
./configure --enable-ort --enable-ort-backend=auto --with-ort-onnx --enable-ort-pool
make
```

### Minimal Build (No Optimizations, No Dependencies)
```bash
./configure --enable-ort --enable-ort-backend=none --disable-ort-pool
make
```

## Troubleshooting

### Common Issues

1. **Backend not available**: Ensure your compiler and system support the requested ISA extension
3. **WASM build fails**: See [https://github.com/krakjoe/em](krakjoe/em)
4. **Multiple backends conflict**: Only one backend can be enabled at a time

### Verification

During configure, the extension will report:
- Detected ISA extensions
- Selected backend configuration
- ONNX Runtime status
- Pooling configuration

## Build Flow Overview

```mermaid
flowchart TD
    A[Start Build] --> B{Enable ORT?<br/>--enable-ort}
    B -->|No| Z[Skip Extension]
    B -->|Yes| C[Detect ISA Extensions]
    
    C --> D[Check WASM SIMD<br/>-msimd128 + wasm_simd128.h]
    C --> E[Check ARM NEON<br/>-march=armv8-a+simd + arm_neon.h]
    C --> F[Check RISCV64<br/>-march=rv64gcv + riscv_vector.h]
    C --> G[Check AVX2<br/>-mavx2 + immintrin.h]
    C --> H[Check SSE4.1<br/>-msse4.1 + smmintrin.h]
    C --> I[Check SSE2<br/>-msse2 + emmintrin.h]
    
    D --> J{Backend Config<br/>--enable-ort-backend}
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J
    
    J -->|auto| K[Auto-Select Best Available]
    J -->|none| L[No Backend Optimization]
    J -->|specific| M[Use Specified Backend]
    
    K --> N{WASM Available?}
    N -->|Yes| O[Use WASM Backend]
    N -->|No| P{NEON Available?}
    P -->|Yes| Q[Use NEON Backend]
    P -->|No| R{RISCV64 Available?}
    R -->|Yes| S[Use RISCV64 Backend]
    R -->|No| T{AVX2 Available?}
    T -->|Yes| U[Use AVX2 Backend]
    T -->|No| V{SSE4.1 Available?}
    V -->|Yes| W[Use SSE4.1 Backend]
    V -->|No| X{SSE2 Available?}
    X -->|Yes| Y[Use SSE2 Backend]
    X -->|No| L
    
    M --> AA{Validate Backend<br/>Availability}
    AA -->|Valid| AB[Configure Backend]
    AA -->|Invalid| AC[Error: Backend Not Available]
    
    O --> AD[Configure WASM Files]
    Q --> AE[Configure NEON Files]
    S --> AF[Configure RISCV64 Files]
    U --> AG[Configure AVX2 Files]
    W --> AH[Configure SSE4.1 Files]
    Y --> AI[Configure SSE2 Files]
    AB --> AJ[Configure Specified Backend Files]
    
    AD --> AK{ONNX Runtime?<br/>--with-ort-onnx}
    AE --> AK
    AF --> AK
    AG --> AK
    AH --> AK
    AI --> AK
    AJ --> AK
    L --> AK
    
    AK -->|Yes| AL[Link ONNX Runtime<br/>libonnxruntime >= 1.16]
    AK -->|No| AM[Skip ONNX Runtime]
    
    AL --> AN{Pooling?<br/>--enable-ort-pool}
    AM --> AN
    
    AN -->|Yes| AO[Enable Thread Pooling]
    AN -->|No| AP[Disable Pooling]
    
    AO --> AQ[Check Atomic Operations<br/>__atomic_add_fetch]
    AP --> AQ
    
    AQ --> AR[Compile Core Sources]
    AR --> AS[Compile Math Operations]
    AS --> AT[Compile Backend Sources]
    AT --> AU[Link Extension]
    AU --> AV[Build Complete]
    
    AD -.-> AW[Note: WASM forces<br/>ext_shared=no<br/>PHP_ORT_POOL=no]
    
    style A fill:#e1f5fe
    style AV fill:#c8e6c9
    style AC fill:#ffcdd2
    style Z fill:#f5f5f5
```
