import numpy as np

types = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32,
    np.float32, np.float64
]

values = np.array([-3, -2, -1, 0, 1, 2, 3])

for t in types:
    arr = values.astype(t)
    print(f"\nType: {t.__name__}")
    for func in [np.round]:
        result = func(arr)
        print(f"{func.__name__}: dtype={result.dtype}, values={result}")
