--TEST--
ORT\Math\sqrt: line-by-line, all types, shape, error handling, numpy semantics
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

// 1. Sqrt for signed/real types: negatives, zero, positives
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ORT\Tensor\Transient([5], [-1, 0, 4, 9, -9], $type);
    $result = ORT\Math\sqrt($a);
    echo "PASS: $name sqrt [-1,0,4,9,-9]\n";
    print_result($result);
}
// 1b. Sqrt for unsigned types: only non-negative values
foreach ($unsigned_types as $name => $type) {
    $a = new ORT\Tensor\Transient([3], [0, 4, 9], $type);
    $result = ORT\Math\sqrt($a);
    echo "PASS: $name sqrt [0,4,9]\n";
    print_result($result);
}


// 2. Sqrt of zeros and ones (valid for all types)
foreach ($types as $name => $type) {
    if ($type == \ORT\Tensor::BOOL) {
        /* meaningless */
        continue;
    }
    $a = new ORT\Tensor\Transient([6], [0,1,0,1,0,1], $type);
    $result = ORT\Math\sqrt($a);
    echo "PASS: $name sqrt zeros/ones\n";
    print_result($result);
}


// 3. Sqrt of 2D tensor for INT8 (with negative and positive values)
$a = new ORT\Tensor\Transient([2,2], [[12,16],[24,-128]], ORT\Tensor::INT8);
$result = ORT\Math\sqrt($a);
echo "PASS: INT8 sqrt 2D [[12,16],[24,-128]]\n";
print_result($result);


// 4. Sqrt of bool tensor
$a = new ORT\Tensor\Transient([2,2], [[true,false],[false,true]], ORT\Tensor::BOOL);
$result = ORT\Math\sqrt($a);
echo "PASS: BOOL sqrt [bool 2x2] (numpy semantics)\n";
print_result($result);


// 5. Sqrt of large and small values (real types only)
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([4], [1e10, 1e-10, 0, 1], $type);
    $result = ORT\Math\sqrt($a);
    echo "PASS: $name sqrt extremes\n";
    print_result($result);
}


// 6. Sqrt of empty tensor (should error)
try {
    $a = new ORT\Tensor\Transient([0], [], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: FLOAT
SHAPE: [5]
PASS: DOUBLE sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: DOUBLE
SHAPE: [5]
PASS: INT8 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: FLOAT
SHAPE: [5]
PASS: INT16 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: FLOAT
SHAPE: [5]
PASS: INT32 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: DOUBLE
SHAPE: [5]
PASS: INT64 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: DOUBLE
SHAPE: [5]
PASS: UINT8 sqrt [0,4,9]
RESULT: %s
TYPE: FLOAT
SHAPE: [3]
PASS: UINT16 sqrt [0,4,9]
RESULT: %s
TYPE: FLOAT
SHAPE: [3]
PASS: UINT32 sqrt [0,4,9]
RESULT: %s
TYPE: DOUBLE
SHAPE: [3]
PASS: FLOAT sqrt zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [6]
PASS: DOUBLE sqrt zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [6]
PASS: INT8 sqrt zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [6]
PASS: INT16 sqrt zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [6]
PASS: INT32 sqrt zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [6]
PASS: INT64 sqrt zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [6]
PASS: UINT8 sqrt zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [6]
PASS: UINT16 sqrt zeros/ones
RESULT: %s
TYPE: FLOAT
SHAPE: [6]
PASS: UINT32 sqrt zeros/ones
RESULT: %s
TYPE: DOUBLE
SHAPE: [6]
PASS: INT8 sqrt 2D [[12,16],[24,-128]]
RESULT: %s
TYPE: FLOAT
SHAPE: [2,2]
PASS: BOOL sqrt [bool 2x2] (numpy semantics)
RESULT: [[1,0],[0,1]]
TYPE: FLOAT
SHAPE: [2,2]
PASS: FLOAT sqrt extremes
RESULT: %s
TYPE: FLOAT
SHAPE: [4]
PASS: DOUBLE sqrt extremes
RESULT: %s
TYPE: DOUBLE
SHAPE: [4]
PASS: Error on empty tensor: %s
