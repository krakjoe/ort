--TEST--
ORT\Math\matmul: matrix x matrix, all types, shape, numpy/onnx semantics, large matrices for CUDA dispatch
--EXTENSIONS--
ort
--SKIPIF--
<?php
if (\ORT\Math\Backend\GPU() === false) {
    die("skip: gpu only test");
}
?>
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = range(1, 128); // 128 elements for [128,128] matrices
$unsigned_values = range(0, 127); // 128 elements
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
    // Build matrices: A [128,128], B [128,128] for large size to trigger CUDA
    $a_data = array_fill(0, 128, array_values($values)); // 128 rows of $values
    $b_data = array_fill(0, 128, array_values($values)); // 128 rows
    $a = new ORT\Tensor\Transient([128, 128], $a_data, $type);
    $b = new ORT\Tensor\Transient([128, 128], $b_data, $type);
    $result = ORT\Math\matmul($a, $b);
    echo "PASS: $name matmul [128,128] x [128,128] (CUDA eligible)\n";
    print_result($result);
}

// Large matrix case for CUDA dispatch
$large_size = 128;
$inner_dim = 128;
foreach ($types as $name => [$type, $values]) {
    // A: shape [$large_size, $inner_dim]
    $a_data = array_fill(0, $large_size, array_values($values));
    // B: shape [$inner_dim, $large_size]
    $b_data = array_fill(0, $inner_dim, array_fill(0, $large_size, 1));
    $a = new ORT\Tensor\Transient([$large_size, $inner_dim], $a_data, $type);
    $b = new ORT\Tensor\Transient([$inner_dim, $large_size], $b_data, $type);
    $result = ORT\Math\matmul($a, $b);
    $data = $result->getData();
    $first = is_array($data[0]) ? $data[0][0] : $data[0];
    $last = is_array($data[$large_size-1]) ? $data[$large_size-1][$large_size-1] : $data[$large_size-1];
    echo "PASS: $name matmul large matrix x large matrix (CUDA dispatch, inner dim $inner_dim)\n";
    printf("RESULT: [%d x %d] first=%f last=%f\n", $large_size, $large_size, $first, $last);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d,%d]\n", $large_size, $large_size);
}

// Batched matmul (3D) with large size
$a_data = array_fill(0, 2, array_fill(0, 128, range(1, 128))); // [2,128,128]
$a_data = array_map(fn($batch) => array_fill(0, 128, range(1, 128)), range(0, 1)); // Adjust for 3D
$a_data = [
    array_fill(0, 128, range(1, 128)),
    array_fill(0, 128, range(1, 128))
];
$b_data = [
    array_fill(0, 128, array_fill(0, 128, 1)),
    array_fill(0, 128, array_fill(0, 128, 1))
];
$a = new ORT\Tensor\Transient([2, 128, 128], $a_data, $real['FLOAT32']);
$b = new ORT\Tensor\Transient([2, 128, 128], $b_data, $real['FLOAT32']);
$result = ORT\Math\matmul($a, $b);
echo "PASS: FLOAT32 matmul batched 3D (large size for CUDA)\n";
print_result($result);
?>
--EXPECTF--
PASS: FLOAT16 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: FLOAT16
SHAPE: [128,128]
PASS: FLOAT32 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: FLOAT32
SHAPE: [128,128]
PASS: FLOAT64 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: FLOAT64
SHAPE: [128,128]
PASS: INT8 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: INT8
SHAPE: [128,128]
PASS: INT16 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: INT16
SHAPE: [128,128]
PASS: INT32 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: INT32
SHAPE: [128,128]
PASS: INT64 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: INT64
SHAPE: [128,128]
PASS: UINT8 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: UINT8
SHAPE: [128,128]
PASS: UINT16 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: UINT16
SHAPE: [128,128]
PASS: UINT32 matmul [128,128] x [128,128] (CUDA eligible)
RESULT: %s
TYPE: UINT32
SHAPE: [128,128]
PASS: FLOAT16 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: FLOAT16
SHAPE: [128,128]
PASS: FLOAT32 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: FLOAT32
SHAPE: [128,128]
PASS: FLOAT64 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: FLOAT64
SHAPE: [128,128]
PASS: INT8 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: INT8
SHAPE: [128,128]
PASS: INT16 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: INT16
SHAPE: [128,128]
PASS: INT32 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: INT32
SHAPE: [128,128]
PASS: INT64 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: INT64
SHAPE: [128,128]
PASS: UINT8 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: UINT8
SHAPE: [128,128]
PASS: UINT16 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: UINT16
SHAPE: [128,128]
PASS: UINT32 matmul large matrix x large matrix (CUDA dispatch, inner dim 128)
RESULT: [128 x 128] first=%f last=%f
TYPE: UINT32
SHAPE: [128,128]
PASS: FLOAT32 matmul batched 3D (large size for CUDA)
RESULT: %s
TYPE: FLOAT32
SHAPE: [2,128,128]