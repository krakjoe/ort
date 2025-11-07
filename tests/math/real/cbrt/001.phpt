--TEST--
ORT\Math\cbrt: line-by-line, all types, shape, numpy semantics
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Cbrt for signed/real types: negatives, zero, positives (large enough for vectorization)
$large_signed = array_merge(
    range(-16, -1),
    [0],
    range(1, 16)
);
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ORT\Tensor\Transient([count($large_signed)], $large_signed, $type);
    $result = ORT\Math\cbrt($a);
    echo "PASS: $name cbrt large signed [-16..-1,0,1..16]\n";
    print_result($result);
}
// 1b. Cbrt for unsigned types: only non-negative values (large enough for vectorization)
$large_unsigned = range(0, 31);
foreach ($unsigned_types as $name => $type) {
    $a = new ORT\Tensor\Transient([count($large_unsigned)], $large_unsigned, $type);
    $result = ORT\Math\cbrt($a);
    echo "PASS: $name cbrt large unsigned [0..31]\n";
    print_result($result);
}

// 2. Cbrt of ones and zeros (valid for all types, large set)
$zeros_ones = array_fill(0, 32, 0);
for ($i = 0; $i < 32; $i += 2) $zeros_ones[$i] = 1;
foreach ($types as $name => $type) {
    if ($type == \ORT\Tensor::BOOL) {
        /* meaningless */
        continue;
    }
    $a = new ORT\Tensor\Transient([count($zeros_ones)], $zeros_ones, $type);
    $result = ORT\Math\cbrt($a);
    echo "PASS: $name cbrt large zeros/ones\n";
    print_result($result);
}

// 3. Cbrt of 2D tensor for INT8 (with negative and positive values, large enough)
$int8_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i % 2 === 0 ? 1 : -1) * ($i * 8 + $j + 1);
    }
    $int8_2d[] = $row;
}
$a = new ORT\Tensor\Transient([8,8], $int8_2d, ORT\Tensor::INT8);
$result = ORT\Math\cbrt($a);
echo "PASS: INT8 cbrt 2D 8x8 alt sign\n";
print_result($result);

// 4. Cbrt of bool tensor (large enough)
$bool_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i + $j) % 2 === 0;
    }
    $bool_2d[] = $row;
}
$a = new ORT\Tensor\Transient([8,8], $bool_2d, ORT\Tensor::BOOL);
$result = ORT\Math\cbrt($a);
echo "PASS: BOOL cbrt [bool 8x8] (numpy semantics)\n";
print_result($result);

// 5. Cbrt of large and small values (real types only, large set)
$extremes = array_merge([
    -1e10, -1e-10, 0, 1e-10, 1e10, 1, -1
], range(2, 33));
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([count($extremes)], $extremes, $type);
    $result = ORT\Math\cbrt($a);
    echo "PASS: $name cbrt extremes large\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT16 cbrt large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT32 cbrt large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT64 cbrt large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 cbrt large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: INT16 cbrt large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: INT32 cbrt large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 cbrt large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: UINT8 cbrt large unsigned [0..31]
RESULT: %s
TYPE: FLOAT16
SHAPE: [32]
PASS: UINT16 cbrt large unsigned [0..31]
RESULT: %s
TYPE: FLOAT32
SHAPE: [32]
PASS: UINT32 cbrt large unsigned [0..31]
RESULT: %s
TYPE: FLOAT64
SHAPE: [32]
PASS: FLOAT16 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT16
SHAPE: [32]
PASS: FLOAT32 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT32
SHAPE: [32]
PASS: FLOAT64 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT64
SHAPE: [32]
PASS: INT8 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT16
SHAPE: [32]
PASS: INT16 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT32
SHAPE: [32]
PASS: INT32 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT64
SHAPE: [32]
PASS: INT64 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT64
SHAPE: [32]
PASS: UINT8 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT16
SHAPE: [32]
PASS: UINT16 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT32
SHAPE: [32]
PASS: UINT32 cbrt large zeros/ones
RESULT: %s
TYPE: FLOAT64
SHAPE: [32]
PASS: INT8 cbrt 2D 8x8 alt sign
RESULT: %s
TYPE: FLOAT16
SHAPE: [8,8]
PASS: BOOL cbrt [bool 8x8] (numpy semantics)
RESULT: %s
TYPE: FLOAT16
SHAPE: [8,8]
PASS: FLOAT16 cbrt extremes large
RESULT: %s
TYPE: FLOAT16
SHAPE: [39]
PASS: FLOAT32 cbrt extremes large
RESULT: %s
TYPE: FLOAT32
SHAPE: [39]
PASS: FLOAT64 cbrt extremes large
RESULT: %s
TYPE: FLOAT64
SHAPE: [39]
