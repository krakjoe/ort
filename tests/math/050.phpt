--TEST--
ONNX\Math\matmul: type promotion, shape, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;


$types = [
    'FLOAT'   => ONNX\Tensor::FLOAT,
    'DOUBLE'  => ONNX\Tensor::DOUBLE,
    'INT8'    => ONNX\Tensor::INT8,
    'INT16'   => ONNX\Tensor::INT16,
    'INT32'   => ONNX\Tensor::INT32,
    'INT64'   => ONNX\Tensor::INT64,
    'UINT8'   => ONNX\Tensor::UINT8,
    'UINT16'  => ONNX\Tensor::UINT16,
    'UINT32'  => ONNX\Tensor::UINT32,
    // BOOL intentionally omitted: matmul not defined for bool
];

function print_result($result) {
    if ($result instanceof Tensor) {
        echo 'RESULT: ' . json_encode($result->getData()) . "\n";
        echo 'TYPE: ' . $result->getType() . "\n";
        echo 'SHAPE: [' . implode(',', $result->getShape()) . "]\n";
    } else {
        echo "NOTENSOR\n";
    }
}

// 1. Basic matmul for all types (2x3 x 3x2)
foreach ($types as $name => $type) {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    $b = new ONNX\Tensor\Transient([3,2], [[7,8],[9,10],[11,12]], $type);
    try {
        $result = ONNX\Math\matmul($a, $b);
        echo "PASS: $name matmul $name\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name matmul $name: ".$e->getMessage()."\n";
    }
}

// 2. Type promotion (int8 x uint8, int32 x float, float x double, etc.)
$promotions = [
    ['A' => 'INT8',   'B' => 'UINT8',  'expect_type' => Tensor::INT32],
    ['A' => 'INT32',  'B' => 'FLOAT',  'expect_type' => Tensor::FLOAT],
    ['A' => 'FLOAT',  'B' => 'DOUBLE', 'expect_type' => Tensor::DOUBLE],
    ['A' => 'INT16',  'B' => 'UINT16', 'expect_type' => Tensor::INT32],
    // ['A' => 'UINT32', 'B' => 'INT64',  'expect_type' => Tensor::INT64], // Not supported by ONNX/NumPy promotion
];
foreach ($promotions as $case) {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $types[$case['A']]);
    $b = new ONNX\Tensor\Transient([3,2], [[7,8],[9,10],[11,12]], $types[$case['B']]);
    try {
        $result = ONNX\Math\matmul($a, $b);
        $type = $result->getType();
        echo "PROMOTE: {$case['A']} x {$case['B']} => type $type\n";
        if ($type !== $case['expect_type']) {
            echo "FAIL: Promotion result type mismatch\n";
        }
    } catch (Throwable $e) {
        echo "FAIL: {$case['A']} x {$case['B']}: ".$e->getMessage()."\n";
    }
}

// 3. Error: incompatible shapes
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ONNX\Tensor::FLOAT);
    $b = new ONNX\Tensor\Transient([4,2], [[1,2],[3,4],[5,6],[7,8]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\matmul($a, $b);
    echo "FAIL: Did not throw on incompatible shapes\n";
} catch (Throwable $e) {
    echo "PASS: Error on incompatible shapes: ".$e->getMessage()."\n";
}

// 4. Error: unsupported type promotion (UINT32 x INT64)
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ONNX\Tensor::UINT32);
    $b = new ONNX\Tensor\Transient([3,2], [[7,8],[9,10],[11,12]], ONNX\Tensor::INT64);
    $result = ONNX\Math\matmul($a, $b);
    echo "FAIL: Did not throw on unsupported promotion\n";
} catch (Throwable $e) {
    echo "PASS: Error on unsupported promotion: ".$e->getMessage()."\n";
}

// 5. Batched matmul (3D tensors)
// Batched matmul (3D tensors)
$a = new ONNX\Tensor\Transient([2,2,3], [
    [[1,2,3],[4,5,6]],
    [[7,8,9],[10,11,12]]
], ONNX\Tensor::FLOAT);
$b = new ONNX\Tensor\Transient([2,3,2], [
    [[13,14],[15,16],[17,18]],
    [[19,20],[21,22],[23,24]]
], ONNX\Tensor::FLOAT);
try {
    $result = ONNX\Math\matmul($a, $b);
    echo "PASS: Batched matmul\n";
    print_result($result);
} catch (Throwable $e) {
    echo "FAIL: Batched matmul: ".$e->getMessage()."\n";
}

// 6. Error: batch dimension mismatch
try {
    $a = new ONNX\Tensor\Transient([2,2,3], [
        [[1,2,3],[4,5,6]],
        [[7,8,9],[10,11,12]]
    ], ONNX\Tensor::FLOAT);
    $b = new ONNX\Tensor\Transient([3,3,2], [
        [[1,2],[3,4],[5,6]],
        [[7,8],[9,10],[11,12]],
        [[13,14],[15,16],[17,18]]
    ], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\matmul($a, $b);
    echo "FAIL: Did not throw on batch dimension mismatch\n";
} catch (Throwable $e) {
    echo "PASS: Error on batch dimension mismatch: ".$e->getMessage()."\n";
}

// 7. Error: less than 2 dimensions
try {
    $a = new ONNX\Tensor\Transient([3], [1,2,3], ONNX\Tensor::FLOAT);
    $b = new ONNX\Tensor\Transient([3], [4,5,6], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\matmul($a, $b);
    echo "FAIL: Did not throw on <2D tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on <2D tensor: ".$e->getMessage()."\n";
}

// 8. BOOL type: should work (logical matmul)
// BOOL type: should error (not supported)
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,0,1],[0,1,0]], ONNX\Tensor::BOOL);
    $b = new ONNX\Tensor\Transient([3,2], [[1,1],[0,0],[1,1]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\matmul($a, $b);
    echo "FAIL: BOOL matmul did not error\n";
} catch (Throwable $e) {
    echo "PASS: BOOL matmul error: ".$e->getMessage()."\n";
}

?>
--EXPECT--
PASS: FLOAT matmul FLOAT
RESULT: [[58,64],[139,154]]
TYPE: 1
SHAPE: [2,2]
PASS: DOUBLE matmul DOUBLE
RESULT: [[58,64],[139,154]]
TYPE: 11
SHAPE: [2,2]
PASS: INT8 matmul INT8
RESULT: [[58,64],[127,127]]
TYPE: 3
SHAPE: [2,2]
PASS: INT16 matmul INT16
RESULT: [[58,64],[139,154]]
TYPE: 5
SHAPE: [2,2]
PASS: INT32 matmul INT32
RESULT: [[58,64],[139,154]]
TYPE: 6
SHAPE: [2,2]
PASS: INT64 matmul INT64
RESULT: [[58,64],[139,154]]
TYPE: 7
SHAPE: [2,2]
PASS: UINT8 matmul UINT8
RESULT: [[58,64],[139,154]]
TYPE: 2
SHAPE: [2,2]
PASS: UINT16 matmul UINT16
RESULT: [[58,64],[139,154]]
TYPE: 4
SHAPE: [2,2]
PASS: UINT32 matmul UINT32
RESULT: [[58,64],[139,154]]
TYPE: 12
SHAPE: [2,2]
PROMOTE: INT8 x UINT8 => type 6
PROMOTE: INT32 x FLOAT => type 1
PROMOTE: FLOAT x DOUBLE => type 11
PROMOTE: INT16 x UINT16 => type 6
PASS: Error on incompatible shapes: matmul: incompatible matrix dimensions (2, 3) x (4, 2)
PASS: Error on unsupported promotion: matmul: unsupported type promotion (UINT32 x INT64)
PASS: Batched matmul
RESULT: [[[94,100],[229,244]],[[508,532],[697,730]]]
TYPE: 1
SHAPE: [2,2,2]
PASS: Error on batch dimension mismatch: matmul: batch dimension 0 mismatch (2 vs 3)
PASS: Error on <2D tensor: matmul: tensors must have at least 2 dimensions
PASS: BOOL matmul error: validation of data according to the shape provided has failed
