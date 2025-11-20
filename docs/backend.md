# Backends

## CPU Backends

Single-Instruction Multiple Data (SIMD) instructions allow performing operations on multiple values simultaneously within a single CPU instruction using vectorization. This can significantly speed up computations involving large tensors.

`php-ort` uses various backends to accelerate some math functions for the frontend (exposed as `ORT\Math`), this is transparent and (mostly) automatic by default.

By default `php-ort` will attempt to build with the best CPU backend available for your system, selected in this order:

  - WASM
  - NEON
  - RISCV64
  - AVX512
  - AVX2
  - SSE4.1
  - SSE2

To create a build that uses no intrinsics at all, use `--disable-ort-backend`.

To create a build using specific CPU intrinsics, use `--enable-ort-backend` with:

  - wasm
  - neon
  - riscv64
  - avx512
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

**Note: When building a GPU backend the CPU backend is not automatically selected, use cuda,auto for automatic CPU backend detection**

# Acceleration Tables

**Legend:**
- 🟩 = Implemented with ISA intrinsic  (native)
- 🟨 = Implemented with F16V extension (native)
- 🟪 = Implemented with F16C extension (efficient workaround, convert to float32 for compute)
- 🟧 = Not yet implemented, but native or efficient workaround possible and planned
- 🟦 = Not supported by the instruction set (no native support, no efficient workaround)
- ⬛ = Not applicable (mathematically meaningless, e.g. abs(unsigned))

## WASM Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟦    |   🟩    |   🟦    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟦    |   🟩    |   🟦    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟦    |   🟩    |   🟦    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **div**    |   🟦    |   🟩    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟦    |   🟩    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟦    |   🟩    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟦    |   🟩    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟦    |   🟩    |   🟦    |   🟩   |   🟩    |   🟩    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟦    |   🟩    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟦    |   🟩    |   🟦    |   🟩   |   🟩    |   🟩    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟦    |   🟩    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟦    |   🟩    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟦    |   🟩    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟦    |   🟧    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **matmul** |   🟦    |   🟩    |   🟦    |   🟦   |   🟦    |   🟩    |   🟦    |   🟦    |    🟦    |    🟩    |
| **sum**    |   🟦    |   🟧    |   🟦    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **min**    |   🟦    |   🟧    |   🟦    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **max**    |   🟦    |   🟧    |   🟦    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmin** |   🟦    |   🟧    |   🟦    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmax** |   🟦    |   🟧    |   🟦    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **mean**   |   🟦    |   🟧    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟦    |   🟧    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

## NEON Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟨    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **div**    |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **matmul** |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **sum**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **min**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **max**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmin** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmax** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **mean**   |   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

## RISCV64 Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **div**    |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟦    |   🟦    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟦    |   🟦    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟧    |   🟧    |   🟧    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟧    |   🟧    |   🟧    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **matmul** |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **sum**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |    🟧    |    🟧    |
| **min**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |    🟧    |    🟧    |
| **max**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |    🟧    |    🟧    |
| **argmin** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |    🟧    |    🟧    |
| **argmax** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟧    |   🟦    |    🟧    |    🟧    |
| **mean**   |   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

## AVX512 Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟪    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟪    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **div**    |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **matmul** |   🟪    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **sum**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **min**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **max**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmin** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmax** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **mean**   |   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

## AVX2 Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟪    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟪    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **div**    |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟪    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **matmul** |   🟪    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **sum**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **min**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **max**    |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmin** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmax** |   🟧    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **mean**   |   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟧    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

## SSE4.1 Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟦    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟦    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟦    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **div**    |   🟦    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟦    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟦    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟦    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟦    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟦    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **matmul** |   🟦    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **sum**    |   🟦    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **min**    |   🟦    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **max**    |   🟦    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmin** |   🟦    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **argmax** |   🟦    |   🟧    |   🟧    |   🟦   |   🟧    |   🟧    |   🟦    |   🟦    |    🟧    |    🟧    |
| **mean**   |   🟦    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟦    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

## SSE2 Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟦    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟦    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟦    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **div**    |   🟦    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   ⬛    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟦    |   🟦    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟦    |   🟦    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟦    |   🟦    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟦    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟦    |   🟦    |   🟦    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟦    |   🟧    |   🟧    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **matmul** |   🟦    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **sum**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **min**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **max**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **argmin** |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **argmax** |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **mean**   |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

## CUDA Backend

| Operation  | float16 | float32 | float64 |  int8  |  int16  |  int32  |  int64  |  uint8  |  uint16  |  uint32  |
|------------|:-------:|:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **add**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **sub**    |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟩    |   🟩    |    🟩    |    🟩    |
| **mul**    |   🟨    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **div**    |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **mod**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **pow**    |   🟪    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **ceil**   |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **floor**  |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **round**  |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **abs**    |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **sqrt**   |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **neg**    |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **recip**  |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **sign**   |   🟨    |   🟩    |   🟩    |   🟦   |   🟦    |   🟦    |   🟦    |   ⬛    |    ⬛    |    ⬛    |
| **trunc**  |   🟨    |   🟩    |   🟩    |   ⬛   |   ⬛    |   ⬛    |   ⬛    |   ⬛    |    ⬛    |    ⬛    |
| **dot**    |   🟨    |   🟩    |   🟩    |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |    🟩    |    🟩    |
| **matmul** |   🟨    |   🟩    |   🟩    |   🟩   |   🟩    |   🟩    |   🟦    |   🟩    |    🟩    |    🟩    |
| **sum**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **min**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **max**    |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **argmin** |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **argmax** |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **mean**   |   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |
| **softmax**|   🟦    |   🟦    |   🟦    |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |    🟦    |    🟦    |

**Notes:**
- RISCV64 backend uses the RISCV64 Vector Extension (RVV) 1.0 specification with variable-length vectors.
- CUDA uses GPU kernels with managed memory and utilize cuBLAS for matrix operations where applicable, float16 is native.
- NEON F16V (+fp16) vector extensions are optional, support will be detected at build time.
- There are machines with AVX512 F16V support, but they are bleeding edge and effectively inaccessible to developers, for this reason AVX512 float16 support will be implemented using F16C extensions.