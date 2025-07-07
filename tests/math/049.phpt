--TEST--
Test ONNX\Math\dot for all supported types, promotions, and error conditions
--EXTENSIONS--
ort
--FILE--
<?php
// Comprehensive test for ONNX\Math\dot

echo "=== Testing dot product ===\n";

$types = [
    'INT8'   => ONNX\Tensor::INT8,
    'UINT8'  => ONNX\Tensor::UINT8,
    'INT16'  => ONNX\Tensor::INT16,
    'UINT16' => ONNX\Tensor::UINT16,
    'INT32'  => ONNX\Tensor::INT32,
    'UINT32' => ONNX\Tensor::UINT32,
    'INT64'  => ONNX\Tensor::INT64,
    // 'UINT64' not supported
    'FLOAT'  => ONNX\Tensor::FLOAT,
    'DOUBLE' => ONNX\Tensor::DOUBLE,
];

// 1. Test dot for all supported types (self dot)
foreach ($types as $name => $type) {
    $a = new ONNX\Tensor\Transient([3], [1, 2, 3], $type);
    $b = new ONNX\Tensor\Transient([3], [4, 5, 6], $type);
    $result = ONNX\Math\dot($a, $b);
    echo "PASS: $name dot $name = ".$result->getData()[0]."\n";
    if ($result->getShape() !== [1]) {
        echo "FAIL: $name result shape is not scalar\n";
    }
}

// 2. Test type promotion (int8 x uint8, int32 x float, float x double, etc.)
// Only valid ONNX/NumPy promotions:
$promotions = [
    ['A' => 'INT8',   'B' => 'UINT8',  'expect_type' => ONNX\Tensor::INT32],
    ['A' => 'INT32',  'B' => 'FLOAT',  'expect_type' => ONNX\Tensor::FLOAT],
    ['A' => 'FLOAT',  'B' => 'DOUBLE', 'expect_type' => ONNX\Tensor::DOUBLE],
    ['A' => 'INT16',  'B' => 'UINT16', 'expect_type' => ONNX\Tensor::INT32],
    // ['A' => 'UINT32', 'B' => 'INT64',  'expect_type' => ONNX\Tensor::INT64], // Not supported by ONNX/NumPy promotion
];
foreach ($promotions as $case) {
    $a = new ONNX\Tensor\Transient([3], [1, 2, 3], $types[$case['A']]);
    $b = new ONNX\Tensor\Transient([3], [4, 5, 6], $types[$case['B']]);
    $result = ONNX\Math\dot($a, $b);
    $type = $result->getType();
    echo "PROMOTE: {$case['A']} x {$case['B']} => type $type\n";
    if ($type !== $case['expect_type']) {
        echo "FAIL: Promotion result type mismatch\n";
    }
}
// Test that UINT32 x INT64 raises an error (not supported)
try {
    $a = new ONNX\Tensor\Transient([3], [1, 2, 3], $types['UINT32']);
    $b = new ONNX\Tensor\Transient([3], [4, 5, 6], $types['INT64']);
    $result = ONNX\Math\dot($a, $b);
    echo "FAIL: UINT32 x INT64 did not error\n";
} catch (ONNX\Status\Math\InvalidType $e) {
    echo "PASS: UINT32 x INT64 error\n";
}

// 3. Test error: mismatched lengths
try {
    $a = new ONNX\Tensor\Transient([3], [1, 2, 3], ONNX\Tensor::INT32);
    $b = new ONNX\Tensor\Transient([2], [4, 5], ONNX\Tensor::INT32);
    $result = ONNX\Math\dot($a, $b);
    echo "FAIL: Mismatched lengths did not error\n";
} catch (ONNX\Status\Math\InvalidShape $e) {
    echo "PASS: Mismatched lengths error\n";
}

// 4. Test error: non-1D tensors
try {
    $a = new ONNX\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ONNX\Tensor::INT32);
    $b = new ONNX\Tensor\Transient([2, 2], [[5, 6], [7, 8]], ONNX\Tensor::INT32);
    $result = ONNX\Math\dot($a, $b);
    echo "FAIL: Non-1D tensors did not error\n";
} catch (ONNX\Status\Math\InvalidShape $e) {
    echo "PASS: Non-1D tensor error\n";
}

// 5. Test error: unsupported types (e.g., BOOL)
try {
    $a = new ONNX\Tensor\Transient([3], [true, false, true], ONNX\Tensor::BOOL);
    $b = new ONNX\Tensor\Transient([3], [false, true, false], ONNX\Tensor::BOOL);
    $result = ONNX\Math\dot($a, $b);
    echo "FAIL: BOOL type did not error\n";
} catch (ONNX\Status\Math\InvalidType $e) {
    echo "PASS: BOOL type error\n";
}

echo "Dot product tests completed.\n";
?>
--EXPECTF--
=== Testing dot product ===
PASS: INT8 dot INT8 = %d
PASS: UINT8 dot UINT8 = %d
PASS: INT16 dot INT16 = %d
PASS: UINT16 dot UINT16 = %d
PASS: INT32 dot INT32 = %d
PASS: UINT32 dot UINT32 = %d
PASS: INT64 dot INT64 = %d
PASS: FLOAT dot FLOAT = %f
PASS: DOUBLE dot DOUBLE = %f
PROMOTE: INT8 x UINT8 => type %d
PROMOTE: INT32 x FLOAT => type %d
PROMOTE: FLOAT x DOUBLE => type %d
PROMOTE: INT16 x UINT16 => type %d
PASS: UINT32 x INT64 error
PASS: Mismatched lengths error
PASS: Non-1D tensor error
PASS: BOOL type error
Dot product tests completed.
