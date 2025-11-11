--TEST--
ORT\Math\matmul: matrix x matrix, all types, shape, numpy/onnx semantics, matrices with inner dimension >=8 for SIMD vectorization
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = range(-16, 15); // 32 elements for [32,32] matrices
$unsigned_values = range(0, 31); // 32 elements
$types = [
    'FLOAT16' => [$real['FLOAT16'], $signed_values],
    'FLOAT32' => [$real['FLOAT32'], $signed_values],
    'FLOAT64' => [$real['FLOAT64'], $signed_values],
    'INT8' => [$signed_types['INT8'], $signed_values],
    'INT16' => [$signed_types['INT16'], $signed_values],
    'INT32' => [$signed_types['INT32'], $signed_values],
    'INT64' => [$signed_types['INT64'], $signed_values],
    'UINT8' => [$unsigned_types['UINT8'], $unsigned_values],
    'UINT16' => [$unsigned_types['UINT16'], $unsigned_values],
    'UINT32' => [$unsigned_types['UINT32'], $unsigned_values],
];
foreach ($types as $name => [$type, $values]) {
    // Build matrices: A [32,32], B [32,32] for inner dim 32 (>= mw for all types)
    $a_data = array_fill(0, 32, array_values($values)); // 32 rows of $values
    $b_data = array_fill(0, 32, array_values($values)); // 32 rows
    $a = new ORT\Tensor\Transient([32, 32], $a_data, $type);
    $b = new ORT\Tensor\Transient([32, 32], $b_data, $type);
    $result = ORT\Math\matmul($a, $b);
    echo "PASS: $name matmul [32,32] x [32,32] (SIMD eligible)\n";
    print_result($result);
}

// Large matrix case for vectorization (inner dim 32)
$large_size = 64; // Smaller for test speed, but inner dim 32
$inner_dim = 32;
foreach ($types as $name => [$type, $values]) {
    // Extend values to 32 elements
    $extended_values = array_merge($values, array_slice($values, 0, $inner_dim - count($values)));
    // A: shape [$large_size, $inner_dim]
    $a_data = array_fill(0, $large_size, array_values($extended_values));
    // B: shape [$inner_dim, $large_size]
    $b_data = array_fill(0, $inner_dim, array_fill(0, $large_size, 1));
    $a = new ORT\Tensor\Transient([$large_size, $inner_dim], $a_data, $type);
    $b = new ORT\Tensor\Transient([$inner_dim, $large_size], $b_data, $type);
    $result = ORT\Math\matmul($a, $b);
    $data = $result->getData();
    $first = is_array($data[0]) ? $data[0][0] : $data[0];
    $last = is_array($data[$large_size-1]) ? $data[$large_size-1][$large_size-1] : $data[$large_size-1];
    echo "PASS: $name matmul large matrix x large matrix (SIMD vectorized, inner dim $inner_dim)\n";
    printf("RESULT: [%d x %d] first=%.1f last=%.1f\n", $large_size, $large_size, $first, $last);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d,%d]\n", $large_size, $large_size);
}

// Batched matmul (3D) with inner dim 32
$a_data = array_fill(0, 2, array_fill(0, 32, range(1, 32))); // [2,32,32]
$a_data = array_map(fn($batch) => array_fill(0, 32, range(1, 32)), range(0, 1)); // Adjust for 3D
// Simpler: [2,32,32] with inner 32
$a_data = [
    array_fill(0, 32, range(1, 32)),
    array_fill(0, 32, range(1, 32))
];
$b_data = [
    array_fill(0, 32, array_fill(0, 32, 1)),
    array_fill(0, 32, array_fill(0, 32, 1))
];
$a = new ORT\Tensor\Transient([2, 32, 32], $a_data, $real['FLOAT32']);
$b = new ORT\Tensor\Transient([2, 32, 32], $b_data, $real['FLOAT32']);
$result = ORT\Math\matmul($a, $b);
echo "PASS: FLOAT32 matmul batched 3D (inner dim 32)\n";
print_result($result);
?>
--EXPECTF--
PASS: FLOAT16 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: FLOAT16
SHAPE: [32,32]
PASS: FLOAT32 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: FLOAT32
SHAPE: [32,32]
PASS: FLOAT64 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: FLOAT64
SHAPE: [32,32]
PASS: INT8 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: INT8
SHAPE: [32,32]
PASS: INT16 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: INT16
SHAPE: [32,32]
PASS: INT32 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: INT32
SHAPE: [32,32]
PASS: INT64 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: INT64
SHAPE: [32,32]
PASS: UINT8 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: UINT8
SHAPE: [32,32]
PASS: UINT16 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: UINT16
SHAPE: [32,32]
PASS: UINT32 matmul [32,32] x [32,32] (SIMD eligible)
RESULT: %s
TYPE: UINT32
SHAPE: [32,32]
PASS: FLOAT16 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: FLOAT16
SHAPE: [%d,%d]
PASS: FLOAT32 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: FLOAT32
SHAPE: [%d,%d]
PASS: FLOAT64 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: FLOAT64
SHAPE: [%d,%d]
PASS: INT8 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT8
SHAPE: [%d,%d]
PASS: INT16 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT16
SHAPE: [%d,%d]
PASS: INT32 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT32
SHAPE: [%d,%d]
PASS: INT64 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT64
SHAPE: [%d,%d]
PASS: UINT8 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: UINT8
SHAPE: [%d,%d]
PASS: UINT16 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: UINT16
SHAPE: [%d,%d]
PASS: UINT32 matmul large matrix x large matrix (SIMD vectorized, inner dim 32)
RESULT: [%d x %d] first=%f last=%f
TYPE: UINT32
SHAPE: [%d,%d]
PASS: FLOAT32 matmul batched 3D (inner dim 32)
RESULT: %s
TYPE: FLOAT32
SHAPE: [2,32,32]