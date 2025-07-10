--TEST--
ONNX\Math\reduce\axis\mean: all types, axis, keepdims, negative axis, and error handling
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

// 1. Mean along axis=0
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\reduce\axis\mean($a, 0);
        echo "PASS: $name mean axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name mean axis=0: ".$e->getMessage()."\n";
    }
}

// 2. Mean along axis=1
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\reduce\axis\mean($a, 1);
        echo "PASS: $name mean axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name mean axis=1: ".$e->getMessage()."\n";
    }
}

// 3. Mean along axis=-1 (negative axis, should match axis=1)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\reduce\axis\mean($a, -1);
        echo "PASS: $name mean axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name mean axis=-1: ".$e->getMessage()."\n";
    }
}

// 4. Mean along axis=1, keepdims=true
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\reduce\axis\mean($a, 1, true);
        echo "PASS: $name mean axis=1 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name mean axis=1 keepdims: ".$e->getMessage()."\n";
    }
}

// 5. Error: axis not integer
try {
    $a = new ONNX\Tensor\Transient([3], [1,2,3], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\reduce\axis\mean($a, 'foo');
    echo "FAIL: Did not throw on non-integer axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-integer axis: ".$e->getMessage()."\n";
}

// 6. Error: axis out of range
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\reduce\axis\mean($a, 2);
    echo "FAIL: Did not throw on axis out of range\n";
} catch (Throwable $e) {
    echo "PASS: Error on axis out of range: ".$e->getMessage()."\n";
}

// 7. Error: bool mean with non-binary values
try {
    $a = new ONNX\Tensor\Transient([2,2], [[1,2],[3,4]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\reduce\axis\mean($a, 0);
    echo "FAIL: Did not throw on non-binary bool tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-binary bool tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: DOUBLE mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: INT8 mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: INT16 mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: INT32 mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: INT64 mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: UINT8 mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: UINT16 mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: UINT32 mean axis=0
RESULT: [2.5,3.5,4.5]
TYPE: %d
SHAPE: [3]
PASS: BOOL mean axis=0
RESULT: [0.5,0.5,0.5]
TYPE: %d
SHAPE: [3]
PASS: FLOAT mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: DOUBLE mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT8 mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT16 mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT32 mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT64 mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: UINT8 mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: UINT16 mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: UINT32 mean axis=1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: BOOL mean axis=1
RESULT: [0.66%d,0.33%d]
TYPE: %d
SHAPE: [2]
PASS: FLOAT mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: DOUBLE mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT8 mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT16 mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT32 mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: INT64 mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: UINT8 mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: UINT16 mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: UINT32 mean axis=-1
RESULT: [2,5]
TYPE: %d
SHAPE: [2]
PASS: BOOL mean axis=-1
RESULT: [0.66%d,0.33%d]
TYPE: %d
SHAPE: [2]
PASS: FLOAT mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: DOUBLE mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: INT8 mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: INT16 mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: INT32 mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: INT64 mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT8 mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT16 mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT32 mean axis=1 keepdims
RESULT: [[2],[5]]
TYPE: %d
SHAPE: [2,1]
PASS: BOOL mean axis=1 keepdims
RESULT: [[0.66%d],[0.33%d]]
TYPE: %d
SHAPE: [2,1]
PASS: Error on non-integer axis: %s
PASS: Error on axis out of range: mean: axis 2 is out of bounds for tensor with 2 dimensions
PASS: Error on non-binary bool tensor: validation of data according to the shape provided has failed
