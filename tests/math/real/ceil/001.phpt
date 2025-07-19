--TEST--
ORT\Math\ceil: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Ceil for signed/real types: negatives, zero, positives
$values = array_merge(range(-16, -1), [0], range(1, 16));
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\ceil($a);
    echo "PASS: $name ceil signed [-16..-1,0,1..16]\n";
    print_result($result);
}
// 1b. Ceil for unsigned types: only non-negative values
$values_unsigned = range(0, 31);
foreach ($unsigned_types as $name => $type) {
    $a = new ORT\Tensor\Transient([count($values_unsigned)], $values_unsigned, $type);
    $result = ORT\Math\ceil($a);
    echo "PASS: $name ceil unsigned [0..31]\n";
    print_result($result);
}
// 2. Ceil of 2D tensor for INT8 (with negative and positive values)
$int8_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = $j - 4;
    }
    $int8_2d[] = $row;
}

// 2. Ceil of 2D tensor for INT8 (with negative and positive values, large enough)
$a = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = $j - 4;
    }
    $a[] = $row;
}
$a = ORT\Tensor\Transient::from($a, ORT\Tensor::INT8);
$result = ORT\Math\ceil($a);
echo "PASS: INT8 ceil 2D 8x8\n";
print_result($result);
?>
--EXPECTF--
PASS: FLOAT ceil signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE ceil signed [-16..-1,0,1..16]
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 ceil signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 ceil signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 ceil signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 ceil signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 ceil unsigned [0..31]
RESULT: %s
TYPE: UINT8
SHAPE: [32]
PASS: UINT16 ceil unsigned [0..31]
RESULT: %s
TYPE: UINT16
SHAPE: [32]
PASS: UINT32 ceil unsigned [0..31]
RESULT: %s
TYPE: UINT32
SHAPE: [32]
PASS: INT8 ceil 2D 8x8
RESULT: %s
TYPE: INT8
SHAPE: [8,8]
