--TEST--
ORT\Math\reduce\axis\argmin: all types, axis, keepdims, negative axis, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Argmin along axis=0
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\argmin($a, 0);
        echo "PASS: $name argmin axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name argmin axis=0: ".$e->getMessage()."\n";
    }
}

// 2. Argmin along axis=1
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\argmin($a, 1);
        echo "PASS: $name argmin axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name argmin axis=1: ".$e->getMessage()."\n";
    }
}

// 3. Argmin along axis=-1 (negative axis, should match axis=1)
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\argmin($a, -1);
        echo "PASS: $name argmin axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name argmin axis=-1: ".$e->getMessage()."\n";
    }
}

// 4. Argmin along axis=1, keepdims=true
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\argmin($a, 1, true);
        echo "PASS: $name argmin axis=1 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name argmin axis=1 keepdims: ".$e->getMessage()."\n";
    }
}

// 5. Error: axis not integer
try {
    $a = new ORT\Tensor\Transient([3], [1,2,3], ORT\Tensor::FLOAT32);
    $result = ORT\Math\reduce\axis\argmin($a, 'foo');
    echo "FAIL: Did not throw on non-integer axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-integer axis: ".$e->getMessage()."\n";
}

// 6. Error: axis out of range
try {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\reduce\axis\argmin($a, 2);
    echo "FAIL: Did not throw on axis out of range\n";
} catch (Throwable $e) {
    echo "PASS: Error on axis out of range: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT16 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: FLOAT32 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: FLOAT64 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: INT8 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: INT16 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: INT32 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: INT64 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: UINT8 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: UINT16 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: UINT32 argmin axis=0
RESULT: [0,0,0]
TYPE: INT64
SHAPE: [3]
PASS: BOOL argmin axis=0
RESULT: [0,1,0]
TYPE: INT64
SHAPE: [3]
PASS: FLOAT16 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: FLOAT32 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: FLOAT64 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT8 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT16 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT32 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT64 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: UINT8 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: UINT16 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: UINT32 argmin axis=1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: BOOL argmin axis=1
RESULT: [0,1]
TYPE: INT64
SHAPE: [2]
PASS: FLOAT16 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: FLOAT32 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: FLOAT64 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT8 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT16 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT32 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: INT64 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: UINT8 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: UINT16 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: UINT32 argmin axis=-1
RESULT: [0,0]
TYPE: INT64
SHAPE: [2]
PASS: BOOL argmin axis=-1
RESULT: [0,1]
TYPE: INT64
SHAPE: [2]
PASS: FLOAT16 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: FLOAT32 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: FLOAT64 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: INT8 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: INT16 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: INT32 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: INT64 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: UINT8 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: UINT16 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: UINT32 argmin axis=1 keepdims
RESULT: [[0],[0]]
TYPE: INT64
SHAPE: [2,1]
PASS: BOOL argmin axis=1 keepdims
RESULT: [[0],[1]]
TYPE: INT64
SHAPE: [2,1]
PASS: Error on non-integer axis: %s
PASS: Error on axis out of range: argmin: axis 2 is out of bounds for tensor with 2 dimensions