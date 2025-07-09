# SIMD Backends

Single-Instruction Multiple Data (SIMD) instructions allow performing operations on multiple values simultaneously within a single CPU instruction using vectorization. This can significantly speed up computations involving large tensors.

`php-ort` uses SIMD backends to accelerate some math functions for the frontend (`ONNX\Math`), this is transparent and automatic by default.

By default `php-ort` will attempt to build with the best backend available for your system, selected in this order:

  - AVX2
  - SSE4.1
  - SSE2

To create a build that uses no intrinsics at all, use the configure flag `--disable-ort-simd`.

To create a build using specific intrinsics, use (combinations of) the following flags:

  - To disable AVX2:   `--disable-ort-avx2`  (will results in an SSE4.1 default)
  - To disable SSE4.1: `--disable-ort-sse41` (will result in an SSE2 default)
  - To disable SSE2:   `--disable-ort-sse2`  (will result in no intrinsics)

For example:

For an SSE4.1 build:

```
./configure --disable-ort-avx2
```

For an SSE2 build:

```
./configure --disable-ort-avx2 --disable-ort-sse41
```


# SIMD Acceleration Tables

**Legend:**
- 🟩 = Implemented with SIMD intrinsic in this backend (native, efficient)
- 🟦 = Not supported by the instruction set (no native SIMD intrinsic, only possible with slow or awkward workaround)
- 🟧 = Not yet implemented, but possible and planned (native or efficient workaround exists)
- ⬛ = Not applicable (mathematically meaningless, e.g. abs(unsigned))
- 🟥 = Not planned (would require extreme workaround or is not practical)

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
| **dot**   |  🟧   |   🟧   |   🟦   |   🟦    |   🟦    |   🟦    |   🟦    |   🟦     |   🟦     |
| **matmul**|  🟩   |   🟩   |   🟦   |   🟩    |   🟩    |   🟦    |   🟦    |   🟩     |   🟩     |

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

**Notes:**
- Cells marked ⬛ are mathematically meaningless (e.g. abs or sign of unsigned types).
- Cells marked 🟦 are not supported by the instruction set and would require a slow or complex workaround (e.g. int8_t × int8_t multiply on AVX2).
- Cells marked 🟧 are possible and planned, but not yet implemented.
- Cells marked 🟥 are technically possible but not planned due to impracticality or extreme inefficiency.
