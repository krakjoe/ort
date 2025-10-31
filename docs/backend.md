# Backends

## CPU Backends

Single-Instruction Multiple Data (SIMD) instructions allow performing operations on multiple values simultaneously within a single CPU instruction using vectorization. This can significantly speed up computations involving large tensors.

`php-ort` uses various backends to accelerate some math functions for the frontend (exposed as `ORT\Math`), this is transparent and (mostly) automatic by default.

By default `php-ort` will attempt to build with the best CPU backend available for your system, selected in this order:

  - WASM
  - NEON
  - RISCV64
  - AVX2
  - SSE4.1
  - SSE2

To create a build that uses no intrinsics at all, use `--disable-ort-backend`.

To create a build using specific CPU intrinsics, use `--enable-ort-backend` with:

  - wasm
  - neon
  - riscv64
  - avx2
  - sse41
  - sse2

For example:

For an SSE4.1 build:

```
./configure --enable-ort-backend=sse41
```

For an SSE2 build:

```
./configure --enable-ort-backend=sse2
```

For a NEON build:

```
./configure --enable-ort-backend=neon
```

For a RISCV64 build:

```
./configure --enable-ort-backend=riscv64
```

## GPU Backends

`php-ort` supports a single GPU backend and CPU backend build combination:

Currently implemented GPU backends:

  - cuda

See [gpu.md](gpu.md) for more information about GPU based acceleration.

**Note: When building a GPU backend the CPU backend is not automatically selected, you are required to select a specific CPU backend**

# Acceleration Tables

**Legend:**
- 🟩 = Implemented with ISA intrinsic (or other API) in this backend (native, efficient)
- 🟦 = Not supported by the instruction set (no native SIMD intrinsic, only possible with slow or awkward workaround)
- 🟧 = Not yet implemented, but possible and planned (native or efficient workaround exists)
- ⬛ = Not applicable (mathematically meaningless, e.g. abs(unsigned))
- 🟥 = Not planned (would require extreme workaround or is not practical)

## WASM Backend

| Operation | float | double | int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t |
|-----------|:-----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**   |  🟩   |   🟦   |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |   🟩     |   🟩     |
| **sub**   |  🟩   |   🟦   |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |   🟩     |   🟩     |
| **mul**   |  🟩   |   🟦   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **div**   |  🟩   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **mod**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **pow**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **ceil**  |  🟩   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **floor** |  🟩   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **round** |  🟩   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **abs**   |  🟩   |   🟦   |   🟩   |   🟩    |   🟩    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **sqrt**  |  🟩   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **neg**   |  🟩   |   🟦   |   🟩   |   🟩    |   🟩    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **recip** |  🟩   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **sign**  |  🟩   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **trunc** |  🟩   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **dot**   |  🟧   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **matmul**|  🟩   |   🟦   |   🟦   |   🟦    |   🟩    |   🟦    |   🟦    |   🟦     |   🟩     |
| **sum**   |  🟧   |   🟦   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **min**   |  🟧   |   🟦   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **max**   |  🟧   |   🟦   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **mean**  |  🟧   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **softmax**|  🟧   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |

## NEON Backend

| Operation | float | double | int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t |
|-----------|:-----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **sub**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **mul**   |  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **div**   |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **mod**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **pow**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **ceil**  |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **floor** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **round** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **abs**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |   ⬛     |   ⬛     |
| **sqrt**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **neg**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |   ⬛     |   ⬛     |
| **recip** |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **sign**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **trunc** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **dot**   |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **matmul**|  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |   🟩     |   🟩     |
| **sum**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **min**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **max**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **mean**  |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **softmax**|  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |

## RISCV64 Backend

| Operation | float | double | int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t |
|-----------|:-----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **sub**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **mul**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **div**   |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **mod**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **pow**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **ceil**  |  🟦   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **floor** |  🟦   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **round** |  🟧   |   🟧   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **abs**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |   ⬛     |   ⬛     |
| **sqrt**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **neg**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |   ⬛     |   ⬛     |
| **recip** |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **sign**  |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |   ⬛     |   ⬛     |
| **trunc** |  🟧   |   🟧   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **dot**   |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **matmul**|  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **sum**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |   🟧     |   🟧     |
| **min**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |   🟧     |   🟧     |
| **max**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |   🟧     |   🟧     |
| **mean**  |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **softmax**|  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |

## AVX2 Backend

| Operation | float | double | int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t |
|-----------|:-----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **sub**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **mul**   |  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **div**   |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **mod**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **pow**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **ceil**  |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **floor** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **round** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **abs**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **sqrt**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **neg**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **recip** |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **sign**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **trunc** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **dot**   |  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **matmul**|  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |   🟩     |   🟩     |
| **sum**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **min**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **max**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **mean**  |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **softmax**|  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |

## SSE4.1 Backend

| Operation | float | double | int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t |
|-----------|:-----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **sub**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **mul**   |  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **div**   |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **mod**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **pow**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **ceil**  |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **floor** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **round** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **abs**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **sqrt**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **neg**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **recip** |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **sign**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **trunc** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **dot**   |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **matmul**|  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **sum**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **min**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **max**   |  🟧   |   🟧   |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |   🟧     |   🟧     |
| **mean**  |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **softmax**|  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |

## SSE2 Backend

| Operation | float | double | int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t |
|-----------|:-----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |   🟩     |   🟩     |
| **sub**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |   🟩     |   🟩     |
| **mul**   |  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **div**   |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **mod**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **pow**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |   🟦     |   🟦     |
| **ceil**  |  🟦   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **floor** |  🟦   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **round** |  🟦   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **abs**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **sqrt**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **neg**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **recip** |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **sign**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **trunc** |  🟦   |   🟦   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **dot**   |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **matmul**|  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **sum**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **min**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **max**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **mean**  |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **softmax**|  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |

## CUDA Backend

| Operation | float | double | int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t |
|-----------|:-----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **sub**   |  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |   🟩     |   🟩     |
| **mul**   |  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **div**   |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **mod**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **pow**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **ceil**  |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **floor** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **round** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **abs**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **sqrt**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **neg**   |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **recip** |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **sign**  |  🟩   |   🟩   |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |   ⬛     |   ⬛     |
| **trunc** |  🟩   |   🟩   |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |   ⬛     |   ⬛     |
| **dot**   |  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |
| **matmul**|  🟩   |   🟩   |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |   🟩     |   🟩     |
| **sum**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **min**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **max**   |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **mean**  |  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **softmax**|  🟦   |   🟦   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |

**Notes:**
- Cells marked ⬛ are mathematically meaningless (e.g. abs or sign of unsigned types).
- Cells marked 🟦 are not supported by the instruction set and would require a slow or complex workaround (e.g. int8_t × int8_t multiply on AVX2).
- Cells marked 🟧 are possible and planned, but not yet implemented.
- Cells marked 🟥 are technically possible but not planned due to impracticality or extreme inefficiency.
- RISCV64 backend uses the RISCV64 Vector Extension (RVV) 1.0 specification with variable-length vectors.
- CUDA backend implementations use GPU kernels with managed memory and utilize cuBLAS for matrix operations where applicable.