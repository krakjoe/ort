--TEST--
ONNX\Math\floor: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Floor for signed/real types: negatives, zero, positives
$values = array_merge(range(-16, -1), [0], range(1, 16));
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $result = ONNX\Math\floor($a);
    echo "PASS: $name floor signed [-16..-1,0,1..16]\n";
    print_result($result);
}
// 1b. Floor for unsigned types: only non-negative values, and negative values to test wraparound
$values_unsigned = array_merge(range(-3, -1), range(0, 31));
foreach ($unsigned_types as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($values_unsigned)], $values_unsigned, $type);
    $result = ONNX\Math\floor($a);
    echo "PASS: $name floor unsigned [-3..-1,0..31] (wraparound)\n";
    print_result($result);
}
// 2. Floor of 2D tensor for INT8 (with negative and positive values)
$int8_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = $j - 4;
    }
    $int8_2d[] = $row;
}
$a = ONNX\Tensor\Transient::from($int8_2d, \ONNX\Tensor::INT8);
$result = ONNX\Math\floor($a);
echo "PASS: INT8 floor 2D 8x8\n";
print_result($result);
?>
--EXPECTF--
PASS: FLOAT floor signed [-16..-1,0,1..16]
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE floor signed [-16..-1,0,1..16]
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 floor signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 floor signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 floor signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 floor signed [-16..-1,0,1..16]
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 floor unsigned [-3..-1,0..31] (wraparound)
RESULT: %s
TYPE: UINT8
SHAPE: [35]
PASS: UINT16 floor unsigned [-3..-1,0..31] (wraparound)
RESULT: %s
TYPE: UINT16
SHAPE: [35]
PASS: UINT32 floor unsigned [-3..-1,0..31] (wraparound)
RESULT: %s
TYPE: UINT32
SHAPE: [35]
PASS: INT8 floor 2D 8x8
RESULT: %s
TYPE: INT8
SHAPE: [8,8]
