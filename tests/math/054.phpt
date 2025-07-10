--TEST--
ONNX\Math\softmax: type promotion, shape, axis, keepdims, and error handling
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

// 1. Basic softmax for all types (2x3)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\softmax($a);
        echo "PASS: $name softmax\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax: ".$e->getMessage()."\n";
    }
}

// 2. Softmax along axis=0
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\softmax($a, 0);
        echo "PASS: $name softmax axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=0: ".$e->getMessage()."\n";
    }
}

// 3. Softmax along axis=1
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\softmax($a, 1);
        echo "PASS: $name softmax axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=1: ".$e->getMessage()."\n";
    }
}

// 3b. Softmax along axis=-1 (negative axis, should match axis=1)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\softmax($a, -1);
        echo "PASS: $name softmax axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=-1: ".$e->getMessage()."\n";
    }
}

// 4. Softmax along axis=1, keepdims=true
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\softmax($a, 1, true);
        echo "PASS: $name softmax axis=1 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name softmax axis=1 keepdims: ".$e->getMessage()."\n";
    }
}

// 5. 1D softmax
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([3], [true,false,true], $type);
    } else {
        $a = new ONNX\Tensor\Transient([3], [1,2,3], $type);
    }
    try {
        $result = ONNX\Math\softmax($a);
        echo "PASS: $name 1D softmax\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name 1D softmax: ".$e->getMessage()."\n";
    }
}

// 6. Error: empty tensor
try {
    $a = new ONNX\Tensor\Transient([0], [], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\softmax($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}

// 7. Error: axis not integer
try {
    $a = new ONNX\Tensor\Transient([3], [1,2,3], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\softmax($a, 'foo');
    echo "FAIL: Did not throw on non-integer axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-integer axis: ".$e->getMessage()."\n";
}

// 8. Error: axis out of range
try {
    $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\softmax($a, 2);
    echo "FAIL: Did not throw on axis out of range\n";
} catch (Throwable $e) {
    echo "PASS: Error on axis out of range: ".$e->getMessage()."\n";
}

// 9. Error: bool softmax with non-binary values (should not happen, but check)
try {
    $a = new ONNX\Tensor\Transient([2,2], [[1,2],[3,4]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\softmax($a);
    echo "FAIL: Did not throw on non-binary bool tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-binary bool tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT softmax
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: DOUBLE softmax
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: INT8 softmax
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT16 softmax
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT32 softmax
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT64 softmax
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT8 softmax
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT16 softmax
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT32 softmax
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: BOOL softmax
RESULT: [[true,true,true],[true,true,true]]
TYPE: %d
SHAPE: [2,3]
PASS: FLOAT softmax axis=0
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: DOUBLE softmax axis=0
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: INT8 softmax axis=0
RESULT: [[0,0,0],[1,1,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT16 softmax axis=0
RESULT: [[0,0,0],[1,1,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT32 softmax axis=0
RESULT: [[0,0,0],[1,1,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT64 softmax axis=0
RESULT: [[0,0,0],[1,1,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT8 softmax axis=0
RESULT: [[0,0,0],[1,1,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT16 softmax axis=0
RESULT: [[0,0,0],[1,1,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT32 softmax axis=0
RESULT: [[0,0,0],[1,1,1]]
TYPE: %d
SHAPE: [2,3]
PASS: BOOL softmax axis=0
RESULT: [[true,true,true],[true,true,true]]
TYPE: %d
SHAPE: [2,3]
PASS: FLOAT softmax axis=1
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: DOUBLE softmax axis=1
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: INT8 softmax axis=1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT16 softmax axis=1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT32 softmax axis=1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT64 softmax axis=1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT8 softmax axis=1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT16 softmax axis=1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT32 softmax axis=1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: BOOL softmax axis=1
RESULT: [[true,true,true],[true,true,true]]
TYPE: %d
SHAPE: [2,3]
PASS: FLOAT softmax axis=-1
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: DOUBLE softmax axis=-1
RESULT: [[%f,%f,%f],[%f,%f,%f]]
TYPE: %d
SHAPE: [2,3]
PASS: INT8 softmax axis=-1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT16 softmax axis=-1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT32 softmax axis=-1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: INT64 softmax axis=-1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT8 softmax axis=-1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT16 softmax axis=-1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: UINT32 softmax axis=-1
RESULT: [[0,0,1],[0,0,1]]
TYPE: %d
SHAPE: [2,3]
PASS: BOOL softmax axis=-1
RESULT: [[true,true,true],[true,true,true]]
TYPE: %d
SHAPE: [2,3]
PASS: FLOAT softmax axis=1 keepdims
RESULT: [[%f],[%f]]
TYPE: %d
SHAPE: [2,1]
PASS: DOUBLE softmax axis=1 keepdims
RESULT: [[%f],[%f]]
TYPE: %d
SHAPE: [2,1]
PASS: INT8 softmax axis=1 keepdims
RESULT: [[0],[0]]
TYPE: %d
SHAPE: [2,1]
PASS: INT16 softmax axis=1 keepdims
RESULT: [[0],[0]]
TYPE: %d
SHAPE: [2,1]
PASS: INT32 softmax axis=1 keepdims
RESULT: [[0],[0]]
TYPE: %d
SHAPE: [2,1]
PASS: INT64 softmax axis=1 keepdims
RESULT: [[0],[0]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT8 softmax axis=1 keepdims
RESULT: [[0],[0]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT16 softmax axis=1 keepdims
RESULT: [[0],[0]]
TYPE: %d
SHAPE: [2,1]
PASS: UINT32 softmax axis=1 keepdims
RESULT: [[0],[0]]
TYPE: %d
SHAPE: [2,1]
PASS: BOOL softmax axis=1 keepdims
RESULT: [[true],[true]]
TYPE: %d
SHAPE: [2,1]
PASS: FLOAT 1D softmax
RESULT: [%f,%f,%f]
TYPE: %d
SHAPE: [3]
PASS: DOUBLE 1D softmax
RESULT: [%f,%f,%f]
TYPE: %d
SHAPE: [3]
PASS: INT8 1D softmax
RESULT: [0,0,1]
TYPE: %d
SHAPE: [3]
PASS: INT16 1D softmax
RESULT: [0,0,1]
TYPE: %d
SHAPE: [3]
PASS: INT32 1D softmax
RESULT: [0,0,1]
TYPE: %d
SHAPE: [3]
PASS: INT64 1D softmax
RESULT: [0,0,1]
TYPE: %d
SHAPE: [3]
PASS: UINT8 1D softmax
RESULT: [0,0,1]
TYPE: %d
SHAPE: [3]
PASS: UINT16 1D softmax
RESULT: [0,0,1]
TYPE: %d
SHAPE: [3]
PASS: UINT32 1D softmax
RESULT: [0,0,1]
TYPE: %d
SHAPE: [3]
PASS: BOOL 1D softmax
RESULT: [true,true,true]
TYPE: %d
SHAPE: [3]
PASS: Error on empty tensor: shape information must be an array of positive integers
PASS: Error on non-integer axis: softmax: axis must be an integer
PASS: Error on axis out of range: softmax: axis 2 is out of bounds for tensor with 2 dimensions
PASS: Error on non-binary bool tensor: validation of data according to the shape provided has failed
