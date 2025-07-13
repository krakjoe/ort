--TEST--
ONNX\Math\exp2: line-by-line, all types, shape, error handling, numpy semantics
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Exp2 for signed/real types: negatives, zero, positives (large enough for vectorization)
$large_signed = array_merge(
    range(-16, -1),
    [0],
    range(1, 16)
);
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($large_signed)], $large_signed, $type);
    $result = ONNX\Math\exp2($a);
    echo "PASS: $name exp2 large signed [-16..-1,0,1..16]\n";
    print_result($result);
}
// 1b. Exp2 for unsigned types: only non-negative values (large enough for vectorization)
$large_unsigned = range(0, 31);
foreach ($unsigned_types as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($large_unsigned)], $large_unsigned, $type);
    $result = ONNX\Math\exp2($a);
    echo "PASS: $name exp2 large unsigned [0..31]\n";
    print_result($result);
}

// 2. Exp2 of ones and zeros (valid for all types, large set)
$zeros_ones = array_fill(0, 32, 0);
for ($i = 0; $i < 32; $i += 2) $zeros_ones[$i] = 1;
foreach ($types as $name => $type) {
    if ($type == \ONNX\Tensor::BOOL) {
        /* meaningless */
        continue;
    }
    $a = new ONNX\Tensor\Transient([count($zeros_ones)], $zeros_ones, $type);
    $result = ONNX\Math\exp2($a);
    echo "PASS: $name exp2 large zeros/ones\n";
    print_result($result);
}

// 3. Exp2 of 2D tensor for INT8 (with negative and positive values, large enough)
$int8_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i % 2 === 0 ? 1 : -1) * ($i * 8 + $j + 1);
    }
    $int8_2d[] = $row;
}
$a = new ONNX\Tensor\Transient([8,8], $int8_2d, ONNX\Tensor::INT8);
$result = ONNX\Math\exp2($a);
echo "PASS: INT8 exp2 2D 8x8 alt sign\n";
print_result($result);

// 4. Exp2 of bool tensor (large enough)
$bool_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i + $j) % 2 === 0;
    }
    $bool_2d[] = $row;
}
$a = new ONNX\Tensor\Transient([8,8], $bool_2d, ONNX\Tensor::BOOL);
$result = ONNX\Math\exp2($a);
echo "PASS: BOOL exp2 [bool 8x8] (numpy semantics)\n";
print_result($result);

// 5. Exp2 of large and small values (real types only, large set)
$extremes = array_merge([
    1e10, 1e-10, 0, 1
], range(2, 33));
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($extremes)], $extremes, $type);
    $result = ONNX\Math\exp2($a);
    echo "PASS: $name exp2 extremes large\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT exp2 large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE exp2 large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 exp2 large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: INT16 exp2 large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: INT32 exp2 large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT64 exp2 large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: UINT8 exp2 large unsigned [0..31]
RESULT: %s
TYPE: FLOAT
SHAPE: [32]
PASS: UINT16 exp2 large unsigned [0..31]
RESULT: %s
TYPE: FLOAT
SHAPE: [32]
PASS: UINT32 exp2 large unsigned [0..31]
RESULT: %s
TYPE: DOUBLE
SHAPE: [32]
PASS: FLOAT exp2 large zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [32]
PASS: DOUBLE exp2 large zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [32]
PASS: INT8 exp2 large zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [32]
PASS: INT16 exp2 large zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [32]
PASS: INT32 exp2 large zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [32]
PASS: INT64 exp2 large zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [32]
PASS: UINT8 exp2 large zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [32]
PASS: UINT16 exp2 large zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [32]
PASS: UINT32 exp2 large zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [32]
PASS: INT8 exp2 2D 8x8 alt sign
RESULT: %s
TYPE: FLOAT
SHAPE: [8,8]
PASS: BOOL exp2 [bool 8x8] (numpy semantics)
RESULT: %s
TYPE: FLOAT
SHAPE: [8,8]
PASS: FLOAT exp2 extremes large
RESULT: %s
TYPE: FLOAT
SHAPE: [36]
PASS: DOUBLE exp2 extremes large
RESULT: %s
TYPE: DOUBLE
SHAPE: [36]
