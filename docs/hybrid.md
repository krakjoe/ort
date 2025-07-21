# Hybrid Parallelism

In order to saturate the CPU, `ORT\Math` uses a pool of workers (per interpreter[1]). The number of workers will default to the number of available cores. This can be overridden by setting the environment variable `ORT_POOL_CORES` explicitly.

Both vectorized operations (provided by various backends, see `docs/backend.md`) and non-vectorized operations (provided by the frontend where vectorization isn't possible/applicable) are distributed across the pool transparently.

This provides `php-ort` with hybrid parallelism: instruction-level parallelism is provided by the backend, while data-level parallelism is provided by the pool.

For a concrete example, consider this code:

```php
$tensors[] = new \ORT\Tensor\Transient(...);
$tensors[] = new \ORT\Tensor\Transient(...);
ORT\Math\add($tensors[0], $tensors[1]);
```

### Distribution of Computation Across the Pool

Where tensors are large enough for vectorization, the call to `ORT\Math\add` will, on a typical 24-core AVX2 machine, result in:

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

Modern processors use a hybrid layout of performance and efficiency cores; often efficiency cores do not implement AVX2 (or others), and if they execute it, they are doing so via emulation which may be slower than compiler optimized (for release builds) scalar code.

Where `ORT_POOL_CORES` exceeds the number of physical cores available, `ORT_POOL_CORES` will effectively be clamped to the number of physical cores, this is done silently: The reason is that it may be too early in startup to raise errors/warnings gracefully. The alternative, to allow over subscription would create an unpredictable execution environment. This can only be misconfiguration, under normal conditions, omit to set `ORT_POOL_CORES`, the maximum will be detected.

When pools are started, we don't allow the scheduler to decide which core to execute the threads on, we assign them to cores explicitly, and in order: This means that if you set `ORT_POOL_CORES=6`, the first 6 logical processors will be used, only. If you omit to set `ORT_POOL_CORES` we saturate the entire package, and importantly, don't allow the scheduler to run two pool threads on the same core, so while there may be contention for resources (there almost certainly will be, other interpreters have pools too), the contention doesn't come from within that interpreter (it comes from others competing for the same resources).

## [1] Interpreters

In a threaded environment, such as FrankenPHP, an interpreter is a thread.
In a single-threaded environment, such as PHP-FPM, an interpreter is a process.