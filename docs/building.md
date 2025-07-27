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

- **`auto`**: Automatically select the best (as applicable) backend
- **`none`**: Disable backend optimizations
- **`wasm`**: WebAssembly SIMD backend (*nix only)
- **`neon`**: ARM NEON backend (*nix only)
- **`avx2`**: AVX2 backend
- **`sse41`**: SSE4.1 backend
- **`sse2`**: SSE2 backend

## Backend Priority Order

When using `--enable-ort-backend=auto`, backends are selected in this priority order:

1. **WASM** (highest priority)
2. **NEON**
3. **AVX2**
4. **SSE4.1**
5. **SSE2** (lowest priority)

## Special Considerations

### WASM Backend Notes

- Forces `ext_shared=no` (static linking only)
- Disables thread pooling (`PHP_ORT_POOL=no`)
- Requires in-tree builds (source directory must equal build directory)
- See [dist/emsdk](dist/emsdk) for building webasm runtime

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
3. **WASM build fails**: See [dist/emsdk](dist/emsdk)
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
    C --> F[Check AVX2<br/>-mavx2 + immintrin.h]
    C --> G[Check SSE4.1<br/>-msse4.1 + smmintrin.h]
    C --> H[Check SSE2<br/>-msse2 + emmintrin.h]
    
    D --> I{Backend Config<br/>--enable-ort-backend}
    E --> I
    F --> I
    G --> I
    H --> I
    
    I -->|auto| J[Auto-Select Best Available]
    I -->|none| K[No Backend Optimization]
    I -->|specific| L[Use Specified Backend]
    
    J --> M{WASM Available?}
    M -->|Yes| N[Use WASM Backend]
    M -->|No| O{NEON Available?}
    O -->|Yes| P[Use NEON Backend]
    O -->|No| Q{AVX2 Available?}
    Q -->|Yes| R[Use AVX2 Backend]
    Q -->|No| S{SSE4.1 Available?}
    S -->|Yes| T[Use SSE4.1 Backend]
    S -->|No| U{SSE2 Available?}
    U -->|Yes| V[Use SSE2 Backend]
    U -->|No| K
    
    L --> W{Validate Backend<br/>Availability}
    W -->|Valid| X[Configure Backend]
    W -->|Invalid| Y[Error: Backend Not Available]
    
    N --> AA[Configure WASM Files]
    P --> BB[Configure NEON Files]
    R --> CC[Configure AVX2 Files]
    T --> DD[Configure SSE4.1 Files]
    V --> EE[Configure SSE2 Files]
    X --> FF[Configure Specified Backend Files]
    
    AA --> GG{ONNX Runtime?<br/>--with-ort-onnx}
    BB --> GG
    CC --> GG
    DD --> GG
    EE --> GG
    FF --> GG
    K --> GG
    
    GG -->|Yes| HH[Link ONNX Runtime<br/>libonnxruntime >= 1.16]
    GG -->|No| II[Skip ONNX Runtime]
    
    HH --> JJ{Pooling?<br/>--enable-ort-pool}
    II --> JJ
    
    JJ -->|Yes| KK[Enable Thread Pooling]
    JJ -->|No| LL[Disable Pooling]
    
    KK --> MM[Check Atomic Operations<br/>__atomic_add_fetch]
    LL --> MM
    
    MM --> NN[Compile Core Sources]
    NN --> OO[Compile Math Operations]
    OO --> PP[Compile Backend Sources]
    PP --> QQ[Link Extension]
    QQ --> RR[Build Complete]
    
    AA -.-> SS[Note: WASM forces<br/>ext_shared=no<br/>PHP_ORT_POOL=no]
    
    style A fill:#e1f5fe
    style RR fill:#c8e6c9
    style Y fill:#ffcdd2
    style Z fill:#f5f5f5
```
