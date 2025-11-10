import sys
import os
import numpy as np
import time
import json

# Allow function and repeat from env or argv
func = os.environ.get('BENCH_FUNCTION')
if not func:
    if len(sys.argv) > 1:
        func = sys.argv[1]
    else:
        print("Usage: python bench.py <func> <size> [repeat=10] [--json]")
        sys.exit(1)
size = int(os.environ.get('BENCH_SIZE', sys.argv[2] if len(sys.argv) > 2 else 1024))
repeat = int(os.environ.get('BENCH_REPEAT', sys.argv[3] if len(sys.argv) > 3 else 10))

# Output as JSON if requested
as_json = '--json' in sys.argv or os.environ.get('BENCH_JSON', '').lower() in ('1', 'true', 'yes')

# Select numpy function
func_map = {
    'add': np.add, 
    'mul': np.multiply, 
    'sub': np.subtract,
    'div': np.divide}
np_func = func_map.get(func)
if np_func is None:
    print(f"Unknown function: {func}")
    sys.exit(1)

a = np.random.rand(size, size).astype(np.float32)
b = np.random.rand(size, size).astype(np.float32)

if not as_json:
    print(f"{func}([{size} x {size}, {size} x {size}]), {os.environ.get('OMP_NUM_THREADS', '?')} threads")

# Warmup
for i in range(repeat):
    result = np_func(a, b)

results = []
for i in range(repeat):
    start = time.time()
    result = np_func(a, b)
    elapsed = time.time() - start
    results.append(elapsed)
    if not as_json:
        print(f"run {i+1}/{repeat}: time={elapsed:.6f}s")
    
def trimmed_mean(data, trim=1):
    if len(data) <= 2 * trim:
        return sum(data) / len(data)
    data_sorted = sorted(data)
    trimmed = data_sorted[trim:-trim]
    return sum(trimmed) / len(trimmed)

trim = min(1, repeat // 4)  # trim 1 from each end if repeat >= 4, else no trim
mean_time = sum(results) / len(results)
trimmed = trimmed_mean(results, trim)
ops = size * size
mops = ops / mean_time / 1e6
trimmed_mops = ops / trimmed / 1e6

if as_json:
    out = {
        'function': func,
        'size': size,
        'repeat': repeat,
        'times': results,
        'mean_time': mean_time,
        'trimmed_mean_time': trimmed,
        'mops': mops,
        'trimmed_mops': trimmed_mops,
        'sum': float(result.sum()),
        'threads': int(os.environ.get('OMP_NUM_THREADS', -1)),
    }
    print(json.dumps(out))
else:
    print(f"sum={result.sum():.6f}")
    print(f"mean_time={mean_time:.6f}s")
    print(f"trimmed_mean_time={trimmed:.6f}s")
    print(f"mops={mops:.2f}")
    print(f"trimmed_mops={trimmed_mops:.2f}")
