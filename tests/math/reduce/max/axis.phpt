--TEST--
ORT\Math\reduce\axis\max: all types, axis, keepdims, negative axis, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Max along axis=0
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\max($a, 0);
        echo "PASS: $name max axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=0: ".$e->getMessage()."\n";
    }
}

// 2. Max along axis=1
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\max($a, 1);
        echo "PASS: $name max axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=1: ".$e->getMessage()."\n";
    }
}

// 3. Max along axis=-1 (negative axis, should match axis=1)
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\max($a, -1);
        echo "PASS: $name max axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=-1: ".$e->getMessage()."\n";
    }
}

// 4. Max along axis=1, keepdims=true
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\max($a, 1, true);
        echo "PASS: $name max axis=1 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=1 keepdims: ".$e->getMessage()."\n";
    }
}

// 5. Error: axis not integer
try {
    $a = new ORT\Tensor\Transient([3], [1,2,3], ORT\Tensor::FLOAT);
    $result = ORT\Math\reduce\axis\max($a, 'foo');
    echo "FAIL: Did not throw on non-integer axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-integer axis: ".$e->getMessage()."\n";
}

// 6. Error: axis out of range
try {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ORT\Tensor::FLOAT);
    $result = ORT\Math\reduce\axis\max($a, 2);
    echo "FAIL: Did not throw on axis out of range\n";
} catch (Throwable $e) {
    echo "PASS: Error on axis out of range: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT max axis=0
RESULT: [4,5,6]
TYPE: FLOAT
SHAPE: [3]
PASS: DOUBLE max axis=0
RESULT: [4,5,6]
TYPE: DOUBLE
SHAPE: [3]
PASS: INT8 max axis=0
RESULT: [4,5,6]
TYPE: INT8
SHAPE: [3]
PASS: INT16 max axis=0
RESULT: [4,5,6]
TYPE: INT16
SHAPE: [3]
PASS: INT32 max axis=0
RESULT: [4,5,6]
TYPE: INT32
SHAPE: [3]
PASS: INT64 max axis=0
RESULT: [4,5,6]
TYPE: INT64
SHAPE: [3]
PASS: UINT8 max axis=0
RESULT: [4,5,6]
TYPE: UINT8
SHAPE: [3]
PASS: UINT16 max axis=0
RESULT: [4,5,6]
TYPE: UINT16
SHAPE: [3]
PASS: UINT32 max axis=0
RESULT: [4,5,6]
TYPE: UINT32
SHAPE: [3]
PASS: BOOL max axis=0
RESULT: [true,true,true]
TYPE: BOOL
SHAPE: [3]
PASS: FLOAT max axis=1
RESULT: [3,6]
TYPE: FLOAT
SHAPE: [2]
PASS: DOUBLE max axis=1
RESULT: [3,6]
TYPE: DOUBLE
SHAPE: [2]
PASS: INT8 max axis=1
RESULT: [3,6]
TYPE: INT8
SHAPE: [2]
PASS: INT16 max axis=1
RESULT: [3,6]
TYPE: INT16
SHAPE: [2]
PASS: INT32 max axis=1
RESULT: [3,6]
TYPE: INT32
SHAPE: [2]
PASS: INT64 max axis=1
RESULT: [3,6]
TYPE: INT64
SHAPE: [2]
PASS: UINT8 max axis=1
RESULT: [3,6]
TYPE: UINT8
SHAPE: [2]
PASS: UINT16 max axis=1
RESULT: [3,6]
TYPE: UINT16
SHAPE: [2]
PASS: UINT32 max axis=1
RESULT: [3,6]
TYPE: UINT32
SHAPE: [2]
PASS: BOOL max axis=1
RESULT: [true,true]
TYPE: BOOL
SHAPE: [2]
PASS: FLOAT max axis=-1
RESULT: [3,6]
TYPE: FLOAT
SHAPE: [2]
PASS: DOUBLE max axis=-1
RESULT: [3,6]
TYPE: DOUBLE
SHAPE: [2]
PASS: INT8 max axis=-1
RESULT: [3,6]
TYPE: INT8
SHAPE: [2]
PASS: INT16 max axis=-1
RESULT: [3,6]
TYPE: INT16
SHAPE: [2]
PASS: INT32 max axis=-1
RESULT: [3,6]
TYPE: INT32
SHAPE: [2]
PASS: INT64 max axis=-1
RESULT: [3,6]
TYPE: INT64
SHAPE: [2]
PASS: UINT8 max axis=-1
RESULT: [3,6]
TYPE: UINT8
SHAPE: [2]
PASS: UINT16 max axis=-1
RESULT: [3,6]
TYPE: UINT16
SHAPE: [2]
PASS: UINT32 max axis=-1
RESULT: [3,6]
TYPE: UINT32
SHAPE: [2]
PASS: BOOL max axis=-1
RESULT: [true,true]
TYPE: BOOL
SHAPE: [2]
PASS: FLOAT max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: FLOAT
SHAPE: [2,1]
PASS: DOUBLE max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: DOUBLE
SHAPE: [2,1]
PASS: INT8 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: INT8
SHAPE: [2,1]
PASS: INT16 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: INT16
SHAPE: [2,1]
PASS: INT32 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: INT32
SHAPE: [2,1]
PASS: INT64 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: INT64
SHAPE: [2,1]
PASS: UINT8 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: UINT8
SHAPE: [2,1]
PASS: UINT16 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: UINT16
SHAPE: [2,1]
PASS: UINT32 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: UINT32
SHAPE: [2,1]
PASS: BOOL max axis=1 keepdims
RESULT: [[true],[true]]
TYPE: BOOL
SHAPE: [2,1]
PASS: Error on non-integer axis: %s
PASS: Error on axis out of range: max: axis 2 is out of bounds for tensor with 2 dimensions