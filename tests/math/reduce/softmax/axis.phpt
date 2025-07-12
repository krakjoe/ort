--TEST--
ONNX\Math\reduce\axis\softmax: float/double, axis, keepdims, negative axis, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Axis=0, keepdims=0
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ONNX\Math\reduce\axis\softmax($a, 0, false);
        echo "PASS: $name softmax axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=0: ".$e->getMessage()."\n";
    }
}

// 2. Axis=1, keepdims=0
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ONNX\Math\reduce\axis\softmax($a, 1, false);
        echo "PASS: $name softmax axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=1: ".$e->getMessage()."\n";
    }
}

// 3. Axis=0, keepdims=1
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ONNX\Math\reduce\axis\softmax($a, 0, true);
        echo "PASS: $name softmax axis=0 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=0 keepdims: ".$e->getMessage()."\n";
    }
}

// 4. Axis=1, keepdims=1
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ONNX\Math\reduce\axis\softmax($a, 1, true);
        echo "PASS: $name softmax axis=1 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=1 keepdims: ".$e->getMessage()."\n";
    }
}

// 5. Axis=-1 (last axis), keepdims=0
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    try {
        $result = ONNX\Math\reduce\axis\softmax($a, -1, false);
        echo "PASS: $name softmax axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=-1: ".$e->getMessage()."\n";
    }
}

// 6. Error: invalid axis
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\reduce\axis\softmax($a, 2, false);
    echo "FAIL: Did not throw on invalid axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on invalid axis: ".$e->getMessage()."\n";
}

// 7. Error: integer input
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ONNX\Tensor::INT32);
    $result = ONNX\Math\reduce\axis\softmax($a, 1, false);
    echo "FAIL: Did not throw on integer input\n";
} catch (Throwable $e) {
    echo "PASS: Error on integer input: ".$e->getMessage()."\n";
}

// 8. Error: bool input
try {
    $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\reduce\axis\softmax($a, 1, false);
    echo "FAIL: Did not throw on bool input\n";
} catch (Throwable $e) {
    echo "PASS: Error on bool input: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT softmax axis=0
RESULT: %s
TYPE: FLOAT
SHAPE: [3]
PASS: DOUBLE softmax axis=0
RESULT: %s
TYPE: DOUBLE
SHAPE: [3]
PASS: FLOAT softmax axis=1
RESULT: %s
TYPE: FLOAT
SHAPE: [2]
PASS: DOUBLE softmax axis=1
RESULT: %s
TYPE: DOUBLE
SHAPE: [2]
PASS: FLOAT softmax axis=0 keepdims
RESULT: %s
TYPE: FLOAT
SHAPE: [1,3]
PASS: DOUBLE softmax axis=0 keepdims
RESULT: %s
TYPE: DOUBLE
SHAPE: [1,3]
PASS: FLOAT softmax axis=1 keepdims
RESULT: %s
TYPE: FLOAT
SHAPE: [2,1]
PASS: DOUBLE softmax axis=1 keepdims
RESULT: %s
TYPE: DOUBLE
SHAPE: [2,1]
PASS: FLOAT softmax axis=-1
RESULT: %s
TYPE: FLOAT
SHAPE: [2]
PASS: DOUBLE softmax axis=-1
RESULT: %s
TYPE: DOUBLE
SHAPE: [2]
PASS: Error on invalid axis: softmax: axis 2 is out of bounds for tensor with 2 dimensions
PASS: Error on integer input: softmax: unsupported data type for mathematical function
PASS: Error on bool input: softmax: unsupported data type for mathematical function
