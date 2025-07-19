--TEST--
ORT\Math\sign: all types, shape, error handling, numpy/ORT schema compliance
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

// 1. Sign for signed/real types: negatives, zero, positives (large enough for vectorization)
$large_signed = array_merge(
    range(-16, -1),
    [0],
    range(1, 16)
);
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ORT\Tensor\Transient([count($large_signed)], $large_signed, $type);
    $result = ORT\Math\sign($a);
    echo "PASS: $name sign large signed [-16..-1,0,1..16]\n";
    print_result($result);
}
// 1b. Sign for unsigned types: only non-negative values (large enough for vectorization)
$large_unsigned = range(0, 31);
foreach ($unsigned_types as $name => $type) {
    $a = new ORT\Tensor\Transient([count($large_unsigned)], $large_unsigned, $type);
    $result = ORT\Math\sign($a);
    echo "PASS: $name sign large unsigned [0..31]\n";
    print_result($result);
}
// 2. Sign for real types: large and small values
$extremes = array_merge([
    -1e10, -1e-10, 0, 1e-10, 1e10, 0, 1
], range(2, 33));
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([count($extremes)], $extremes, $type);
    $result = ORT\Math\sign($a);
    echo "PASS: $name sign extremes large\n";
    print_result($result);
}
// 3. Sign for 2D tensor for INT8 (with negative and positive values, large enough)
$int8_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i % 2 === 0 ? 1 : -1) * ($i * 8 + $j + 1);
    }
    $int8_2d[] = $row;
}
$a = new ORT\Tensor\Transient([8,8], $int8_2d, ORT\Tensor::INT8);
$result = ORT\Math\sign($a);
echo "PASS: INT8 sign 2D 8x8 alt sign\n";
print_result($result);
// 4. Sign for bool tensor (should error)
$bool_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i + $j) % 2 === 0;
    }
    $bool_2d[] = $row;
}
try {
    $a = new ORT\Tensor\Transient([8,8], $bool_2d, ORT\Tensor::BOOL);
    $result = ORT\Math\sign($a);
    echo "FAIL: BOOL sign [bool 8x8] (should error)\n";
    print_result($result);
} catch (Throwable $e) {
    echo "PASS: BOOL sign [bool 8x8] error as expected\n";
}
?>
--EXPECTF--
PASS: FLOAT sign large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE sign large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 sign large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 sign large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 sign large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 sign large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 sign large unsigned [0..31]
RESULT: %s
TYPE: UINT8
SHAPE: [32]
PASS: UINT16 sign large unsigned [0..31]
RESULT: %s
TYPE: UINT16
SHAPE: [32]
PASS: UINT32 sign large unsigned [0..31]
RESULT: %s
TYPE: UINT32
SHAPE: [32]
PASS: FLOAT sign extremes large
RESULT: %s
TYPE: FLOAT
SHAPE: [39]
PASS: DOUBLE sign extremes large
RESULT: %s
TYPE: DOUBLE
SHAPE: [39]
PASS: INT8 sign 2D 8x8 alt sign
RESULT: %s
TYPE: INT8
SHAPE: [8,8]
PASS: BOOL sign [bool 8x8] error as expected