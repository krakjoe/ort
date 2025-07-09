--TEST--
ONNX\Math\max: type promotion, shape, axis, keepdims, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;

$types = [
    'FLOAT'   => ONNX\Tensor::FLOAT,
    'DOUBLE'  => ONNX\Tensor::DOUBLE,
    'INT8'    => ONNX\Tensor::INT8,
    'INT16'   => ONNX\Tensor::INT16,
    'INT32'   => ONNX\Tensor::INT32,
    'INT64'   => ONNX\Tensor::INT64,
    'UINT8'   => ONNX\Tensor::UINT8,
    'UINT16'  => ONNX\Tensor::UINT16,
    'UINT32'  => ONNX\Tensor::UINT32,
    'BOOL'    => ONNX\Tensor::BOOL,
];

function print_result($result) {
    if ($result instanceof Tensor) {
        echo 'RESULT: ' . json_encode($result->getData()) . "\n";
        echo 'TYPE: ' . $result->getType() . "\n";
        echo 'SHAPE: [' . implode(',', $result->getShape()) . "]\n";
    } else {
        echo "NOTENSOR\n";
    }
}

// 1. Basic max for all types (2x3)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\max($a);
        echo "PASS: $name max\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max: ".$e->getMessage()."\n";
    }
}

// 2. Max along axis=0
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\max($a, 0);
        echo "PASS: $name max axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=0: ".$e->getMessage()."\n";
    }
}

// 3. Max along axis=1
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\max($a, 1);
        echo "PASS: $name max axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=1: ".$e->getMessage()."\n";
    }
}

// 3b. Max along axis=-1 (negative axis, should match axis=1)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\max($a, -1);
        echo "PASS: $name max axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=-1: ".$e->getMessage()."\n";
    }
}

// 4. Max along axis=1, keepdims=true
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\max($a, 1, true);
        echo "PASS: $name max axis=1 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max axis=1 keepdims: ".$e->getMessage()."\n";
    }
}

// 4b. Max along all axes (result is scalar, result_dims == 0), keepdims=false (default)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\max($a, null, false); // reduce all axes, keepdims false
        echo "PASS: $name max all axes scalar\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max all axes scalar: ".$e->getMessage()."\n";
    }
}

// 4c. Max along all axes (result is scalar, result_dims == 0), keepdims=true
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\max($a, null, true); // reduce all axes, keepdims true
        echo "PASS: $name max all axes keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max all axes keepdims: ".$e->getMessage()."\n";
    }
}

// 5. Error: empty tensor
try {
    $a = new ONNX\Tensor\Transient([0], [], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\max($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}

// 6. Error: axis not integer
try {
    $a = new ONNX\Tensor\Transient([3], [1,2,3], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\max($a, 'foo');
    echo "FAIL: Did not throw on non-integer axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-integer axis: ".$e->getMessage()."\n";
}

// 7. Error: axis out of range
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\max($a, 2);
    echo "FAIL: Did not throw on axis out of range\n";
} catch (Throwable $e) {
    echo "PASS: Error on axis out of range: ".$e->getMessage()."\n";
}

// 8. Error: bool max with non-binary values (should not happen, but check)
try {
    $a = new ONNX\Tensor\Transient([2,2], [[1,2],[3,4]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\max($a);
    echo "FAIL: Did not throw on non-binary bool tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-binary bool tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: DOUBLE max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT8 max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT16 max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT32 max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT64 max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT8 max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT16 max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT32 max
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: BOOL max
RESULT: [true]
TYPE: %d
SHAPE: []
PASS: FLOAT max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: DOUBLE max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: INT8 max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: INT16 max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: INT32 max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: INT64 max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: UINT8 max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: UINT16 max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: UINT32 max axis=0
RESULT: [4,5,6]
TYPE: %d
SHAPE: [3]
PASS: BOOL max axis=0
RESULT: [true,true,true]
TYPE: %d
SHAPE: [3]
PASS: FLOAT max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: DOUBLE max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT8 max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT16 max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT32 max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT64 max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: UINT8 max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: UINT16 max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: UINT32 max axis=1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: BOOL max axis=1
RESULT: [true,true]
TYPE: %d
SHAPE: [2]
PASS: FLOAT max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: DOUBLE max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT8 max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT16 max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT32 max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: INT64 max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: UINT8 max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: UINT16 max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: UINT32 max axis=-1
RESULT: [3,6]
TYPE: %d
SHAPE: [2]
PASS: BOOL max axis=-1
RESULT: [true,true]
TYPE: %d
SHAPE: [2]
PASS: FLOAT max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: DOUBLE max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: INT8 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: INT16 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: INT32 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: INT64 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT8 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT16 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT32 max axis=1 keepdims
RESULT: [[3],[6]]
TYPE: %d
SHAPE: [2,1]
PASS: BOOL max axis=1 keepdims
RESULT: [[true],[true]]
TYPE: %d
SHAPE: [2,1]
PASS: FLOAT max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: DOUBLE max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT8 max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT16 max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT32 max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT64 max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT8 max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT16 max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT32 max all axes scalar
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: BOOL max all axes scalar
RESULT: [true]
TYPE: %d
SHAPE: []
PASS: FLOAT max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: DOUBLE max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT8 max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT16 max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT32 max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: INT64 max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT8 max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT16 max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: UINT32 max all axes keepdims
RESULT: [6]
TYPE: %d
SHAPE: []
PASS: BOOL max all axes keepdims
RESULT: [true]
TYPE: %d
SHAPE: []
PASS: Error on empty tensor: shape information must be an array of positive integers
PASS: Error on non-integer axis: max: axis must be an integer
PASS: Error on axis out of range: max: axis 2 is out of bounds for tensor with 2 dimensions
PASS: Error on non-binary bool tensor: validation of data according to the shape provided has failed
