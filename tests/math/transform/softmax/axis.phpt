--TEST--
ORT\Math\transform\axis\softmax: float32/float64, axis, negative axis, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Axis=0
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ORT\Math\transform\axis\softmax($a, 0);
        echo "PASS: $name softmax axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=0: ".$e->getMessage()."\n";
    }
}

// 2. Axis=1
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ORT\Math\transform\axis\softmax($a, 1);
        echo "PASS: $name softmax axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=1: ".$e->getMessage()."\n";
    }
}

// 3. Axis=-1 (last axis)
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ORT\Math\transform\axis\softmax($a, -1);
        echo "PASS: $name softmax axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=-1: ".$e->getMessage()."\n";
    }
}

// 4. Error: invalid axis
try {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\transform\axis\softmax($a, 2);
    echo "FAIL: Did not throw on invalid axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on invalid axis: ".$e->getMessage()."\n";
}

// 5. Error: integer input
try {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ORT\Tensor::INT32);
    $result = ORT\Math\transform\axis\softmax($a, 1);
    echo "FAIL: Did not throw on integer input\n";
} catch (Throwable $e) {
    echo "PASS: Error on integer input: ".$e->getMessage()."\n";
}

// 6. Error: bool input
try {
    $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], ORT\Tensor::BOOL);
    $result = ORT\Math\transform\axis\softmax($a, 1);
    echo "FAIL: Did not throw on bool input\n";
} catch (Throwable $e) {
    echo "PASS: Error on bool input: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT16 softmax axis=0
RESULT: %s
TYPE: FLOAT16
SHAPE: [2,3]
PASS: FLOAT32 softmax axis=0
RESULT: %s
TYPE: FLOAT32
SHAPE: [2,3]
PASS: FLOAT64 softmax axis=0
RESULT: %s
TYPE: FLOAT64
SHAPE: [2,3]
PASS: FLOAT16 softmax axis=1
RESULT: %s
TYPE: FLOAT16
SHAPE: [2,3]
PASS: FLOAT32 softmax axis=1
RESULT: %s
TYPE: FLOAT32
SHAPE: [2,3]
PASS: FLOAT64 softmax axis=1
RESULT: %s
TYPE: FLOAT64
SHAPE: [2,3]
PASS: FLOAT16 softmax axis=-1
RESULT: %s
TYPE: FLOAT16
SHAPE: [2,3]
PASS: FLOAT32 softmax axis=-1
RESULT: %s
TYPE: FLOAT32
SHAPE: [2,3]
PASS: FLOAT64 softmax axis=-1
RESULT: %s
TYPE: FLOAT64
SHAPE: [2,3]
PASS: Error on invalid axis: softmax: axis 2 is out of bounds for tensor with 2 dimensions
PASS: Error on integer input: softmax: unsupported kernel INT32 -> UNKNOWN
PASS: Error on bool input: softmax: unsupported kernel BOOL -> UNKNOWN