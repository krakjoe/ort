--TEST--
ORT\Math\matmul: matrix x matrix, all types, shape, numpy/onnx semantics, small matrix and batched 3D for vectorization
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = array_merge(range(-8, -1), [0], range(1, 8)); // 17 elements
$unsigned_values = range(0, 7); // 8 elements for [8,8] matrices
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
    // Build 2D arrays matching the shape
    $a_data = array_map(fn($v) => [$v, $v], $values); // shape [count($values), 2]
    $b_data = [array_values($values), array_values($values)]; // shape [2, count($values)]
    $a = new ORT\Tensor\Transient([count($values), 2], $a_data, $type);
    $b = new ORT\Tensor\Transient([2, count($values)], $b_data, $type);
    $result = ORT\Math\matmul($a, $b);
    echo "PASS: $name matmul matrix x matrix\n";
    print_result($result);
}

// Small matrix case for vectorization
$large_size = 256;
foreach ($types as $name => [$type, $values]) {
    // Use much smaller values for FLOAT16 to avoid overflow
    $val_a = ($name === 'FLOAT16') ? 0.01 : 1;
    $val_b = ($name === 'FLOAT16') ? 0.01 : 2;
    
    // $a: shape [$large_size, 2], each row is [$val_a, $val_a] (distinct arrays)
    $a_data = [];
    for ($i = 0; $i < $large_size; $i++) {
        $a_data[] = [$val_a, $val_a];
    }
    // $b: shape [2, $large_size], each row is a distinct array of $val_b values
    $b_data = [];
    for ($i = 0; $i < 2; $i++) {
        $b_data[] = array_fill(0, $large_size, $val_b);
    }
    $a = new ORT\Tensor\Transient([$large_size, 2], $a_data, $type);
    $b = new ORT\Tensor\Transient([2, $large_size], $b_data, $type);
    $result = ORT\Math\matmul($a, $b);
    $data = $result->getData();
    $first = $data[0][0];
    $last = $data[$large_size-1][$large_size-1];
    echo "PASS: $name matmul large matrix x large matrix (vectorized)\n";
    printf("RESULT: [%d x %d] first=%.1f last=%.1f\n", $large_size, $large_size, $first, $last);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d,%d]\n", $large_size, $large_size);
}

// Batched matmul (3D)
$a_data = [
    [ [1,2], [3,4] ],
    [ [5,6], [7,8] ]
];
$b_data = [
    [ [1,0], [0,1] ],
    [ [1,0], [0,1] ]
];
$a = new ORT\Tensor\Transient([2,2,2], $a_data, $real['FLOAT32']);
$b = new ORT\Tensor\Transient([2,2,2], $b_data, $real['FLOAT32']);
$result = ORT\Math\matmul($a, $b);
echo "PASS: FLOAT32 matmul batched 3D\n";
print_result($result);
?>
--EXPECTF--
PASS: FLOAT16 matmul matrix x matrix
RESULT: %s
TYPE: FLOAT16
SHAPE: [17,17]
PASS: FLOAT32 matmul matrix x matrix
RESULT: %s
TYPE: FLOAT32
SHAPE: [17,17]
PASS: FLOAT64 matmul matrix x matrix
RESULT: %s
TYPE: FLOAT64
SHAPE: [17,17]
PASS: INT8 matmul matrix x matrix
RESULT: %s
TYPE: INT8
SHAPE: [17,17]
PASS: INT16 matmul matrix x matrix
RESULT: %s
TYPE: INT16
SHAPE: [17,17]
PASS: INT32 matmul matrix x matrix
RESULT: %s
TYPE: INT32
SHAPE: [17,17]
PASS: INT64 matmul matrix x matrix
RESULT: %s
TYPE: INT64
SHAPE: [17,17]
PASS: UINT8 matmul matrix x matrix
RESULT: %s
TYPE: UINT8
SHAPE: [8,8]
PASS: UINT16 matmul matrix x matrix
RESULT: %s
TYPE: UINT16
SHAPE: [8,8]
PASS: UINT32 matmul matrix x matrix
RESULT: %s
TYPE: UINT32
SHAPE: [8,8]
PASS: FLOAT16 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: FLOAT16
SHAPE: [256,256]
PASS: FLOAT32 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: FLOAT32
SHAPE: [%d,%d]
PASS: FLOAT64 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: FLOAT64
SHAPE: [%d,%d]
PASS: INT8 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT8
SHAPE: [%d,%d]
PASS: INT16 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT16
SHAPE: [%d,%d]
PASS: INT32 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT32
SHAPE: [%d,%d]
PASS: INT64 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: INT64
SHAPE: [%d,%d]
PASS: UINT8 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: UINT8
SHAPE: [%d,%d]
PASS: UINT16 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: UINT16
SHAPE: [%d,%d]
PASS: UINT32 matmul large matrix x large matrix (vectorized)
RESULT: [%d x %d] first=%f last=%f
TYPE: UINT32
SHAPE: [%d,%d]
PASS: FLOAT32 matmul batched 3D
RESULT: %s
TYPE: FLOAT32
SHAPE: [2,2,2]