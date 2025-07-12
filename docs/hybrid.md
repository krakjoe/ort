# Hybrid Parallelism

In order to saturate the CPU, `ONNX\Math` uses a pool of workers (per interpreter[1]). The number of workers will default to the number of available cores. This can be overridden by setting the environment variable `ORT_POOL_CORES` explicitly.

Both vectorized operations (provided by various backends, see `docs/backend.md`) and non-vectorized operations (provided by the frontend where vectorization isn't possible/applicable) are distributed across the pool transparently.

This provides `php-ort` with hybrid parallelism: instruction-level parallelism is provided by the backend, while data-level parallelism is provided by the pool.

For a concrete example, consider this code:

```php
$tensors[] = new \ONNX\Tensor\Transient(...);
$tensors[] = new \ONNX\Tensor\Transient(...);
ONNX\Math\add($tensors[0], $tensors[1]);
```

### Distribution of Computation Across the Pool

Where tensors are large enough for vectorization, the call to `ONNX\Math\add` will, on a typical 24-core AVX2 machine, result in:

  - Input tensors are split into 24 chunks (one per core/thread in the pool).
  - Each worker thread processes its chunk independently and in parallel.
  - Within each thread, the backend (e.g., AVX2) processes multiple elements at once using SIMD vector instructions (e.g., 8 floats per AVX2 instruction).
  - This means each core is executing vectorized math on its chunk, while all cores are working simultaneously.
  - The result: all 24 cores are fully utilized, and each core is running vectorized math for maximum throughput.

### Summary

- If the backend vector width is 8 (e.g., AVX2 for float32), then at peak throughput, the CPU can execute 24 (cores) × 8 (SIMD width) = 192 additions simultaneously in a single cycle.

**There can't be a way to execute math on the cpu any faster!**

#### Considerations

This describes a perfect-world scenario, where there is no contention for resources or other bottlenecks (eg. memory bandwidth, cache misses, etc). Threads are not magic sauce, in practice you may need to tune the number of threads used to achieve optimal performance.

The world isn't perfect ... the point of this architecture is not to make sure that 192 additions per cycle is achieved, but rather to ensure that the CPU may be maximally utilized by all means possible.

## [1] Interpreters

In a threaded environment, such as FrankenPHP, an interpreter is a thread.
In a single-threaded environment, such as PHP-FPM, an interpreter is a process.