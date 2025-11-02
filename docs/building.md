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
- **`avx512`**: AVX512 backend (CPU)
- **`avx2`**: AVX2 backend (CPU)
- **`sse41`**: SSE4.1 backend (CPU)
- **`sse2`**: SSE2 backend (CPU)
- **`cuda`**: CUDA backend (GPU, *nix only)

## Backend Priority Order

When using `--enable-ort-backend=auto`, the CPU backend is selected in this priority order:

1. **WASM** (highest priority)
2. **NEON**
3. **RISCV64**
4. **AVX512**
5. **AVX2**
6. **SSE4.1**
7. **SSE2** (lowest priority)

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
    C --> G[Check AVX512<br/>-mavx512f + immintrin.h]
    C --> H[Check AVX2<br/>-mavx2 + immintrin.h]
    C --> I[Check SSE4.1<br/>-msse4.1 + smmintrin.h]
    C --> J[Check SSE2<br/>-msse2 + emmintrin.h]
    
    D --> K{Backend Config<br/>--enable-ort-backend}
    E --> K
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
    
    K -->|auto| L[Auto-Select Best Available]
    K -->|none| M[No Backend Optimization]
    K -->|specific| N[Use Specified Backend]
    
    L --> O{WASM Available?}
    O -->|Yes| P[Use WASM Backend]
    O -->|No| Q{NEON Available?}
    Q -->|Yes| R[Use NEON Backend]
    Q -->|No| S{RISCV64 Available?}
    S -->|Yes| T[Use RISCV64 Backend]
    S -->|No| U{AVX512 Available?}
    U -->|Yes| V[Use AVX512 Backend]
    U -->|No| W{AVX2 Available?}
    W -->|Yes| X[Use AVX2 Backend]
    W -->|No| Y{SSE4.1 Available?}
    Y -->|Yes| ZZ[Use SSE4.1 Backend]
    Y -->|No| AA{SSE2 Available?}
    AA -->|Yes| AB[Use SSE2 Backend]
    AA -->|No| M
    
    N --> BB{Validate Backend<br/>Availability}
    BB -->|Valid| CC[Configure Backend]
    BB -->|Invalid| DD[Error: Backend Not Available]
    
    P --> EE[Configure WASM Files]
    R --> FF[Configure NEON Files]
    T --> GG[Configure RISCV64 Files]
    V --> HH[Configure AVX512 Files]
    X --> II[Configure AVX2 Files]
    ZZ --> JJ[Configure SSE4.1 Files]
    AB --> KK[Configure SSE2 Files]
    CC --> LL[Configure Specified Backend Files]
    
    EE --> MM{ONNX Runtime?<br/>--with-ort-onnx}
    FF --> MM
    GG --> MM
    HH --> MM
    II --> MM
    JJ --> MM
    KK --> MM
    LL --> MM
    M --> MM
    
    MM -->|Yes| NN[Link ONNX Runtime<br/>libonnxruntime >= 1.16]
    MM -->|No| OO[Skip ONNX Runtime]
    
    NN --> PP{Pooling?<br/>--enable-ort-pool}
    OO --> PP
    
    PP -->|Yes| QQ[Enable Thread Pooling]
    PP -->|No| RR[Disable Pooling]
    
    QQ --> SS[Check Atomic Operations<br/>__atomic_add_fetch]
    RR --> SS
    
    SS --> TT[Compile Core Sources]
    TT --> UU[Compile Math Operations]
    UU --> VV[Compile Backend Sources]
    VV --> WW[Link Extension]
    WW --> XX[Build Complete]
    
    EE -.-> YY[Note: WASM forces<br/>ext_shared=no<br/>PHP_ORT_POOL=no]
    
    style A fill:#e1f5fe
    style XX fill:#c8e6c9
    style DD fill:#ffcdd2
    style Z fill:#f5f5f5
```
