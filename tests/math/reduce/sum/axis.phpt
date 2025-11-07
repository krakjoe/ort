--TEST--
ORT\Math\reduce\axis\sum: all types, axis, keepdims, negative axis, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

$types = [
    'FLOAT16'  => ORT\Tensor::FLOAT16,
    'FLOAT32'   => ORT\Tensor::FLOAT32,
    'FLOAT64'  => ORT\Tensor::FLOAT64,
    'INT8'    => ORT\Tensor::INT8,
    'INT16'   => ORT\Tensor::INT16,
    'INT32'   => ORT\Tensor::INT32,
    'INT64'   => ORT\Tensor::INT64,
    /* compliance, sum(unsigned) -> uint64
        we don't support uint64, so cannot
        support these operations with numpy semantics
    'UINT8'   => ORT\Tensor::UINT8,
    'UINT16'  => ORT\Tensor::UINT16,
    'UINT32'  => ORT\Tensor::UINT32,
    */
    'BOOL'    => ORT\Tensor::BOOL,
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

// 1. Axis=0, keepdims=0
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\sum($a, 0, false);
        echo "PASS: $name sum axis=0\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name sum axis=0: ".$e->getMessage()."\n";
    }
}

// 2. Axis=1, keepdims=0
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\sum($a, 1, false);
        echo "PASS: $name sum axis=1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name sum axis=1: ".$e->getMessage()."\n";
    }
}

// 3. Axis=0, keepdims=1
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\sum($a, 0, true);
        echo "PASS: $name sum axis=0 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name sum axis=0 keepdims: ".$e->getMessage()."\n";
    }
}

// 4. Axis=1, keepdims=1
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\sum($a, 1, true);
        echo "PASS: $name sum axis=1 keepdims\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name sum axis=1 keepdims: ".$e->getMessage()."\n";
    }
}

// 5. Axis=-1 (last axis), keepdims=0
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\axis\sum($a, -1, false);
        echo "PASS: $name sum axis=-1\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name sum axis=-1: ".$e->getMessage()."\n";
    }
}

// 6. Error: invalid axis
try {
    $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\reduce\axis\sum($a, 2, false);
    echo "FAIL: Did not throw on invalid axis\n";
} catch (Throwable $e) {
    echo "PASS: Error on invalid axis: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT16 sum axis=0
RESULT: [5,7,9]
TYPE: %d
SHAPE: [3]
PASS: FLOAT32 sum axis=0
RESULT: [5,7,9]
TYPE: %d
SHAPE: [3]
PASS: FLOAT64 sum axis=0
RESULT: [5,7,9]
TYPE: %d
SHAPE: [3]
PASS: INT8 sum axis=0
RESULT: [5,7,9]
TYPE: %d
SHAPE: [3]
PASS: INT16 sum axis=0
RESULT: [5,7,9]
TYPE: %d
SHAPE: [3]
PASS: INT32 sum axis=0
RESULT: [5,7,9]
TYPE: %d
SHAPE: [3]
PASS: INT64 sum axis=0
RESULT: [5,7,9]
TYPE: %d
SHAPE: [3]
PASS: BOOL sum axis=0
RESULT: [1,1,1]
TYPE: %d
SHAPE: [3]
PASS: FLOAT16 sum axis=1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: FLOAT32 sum axis=1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: FLOAT64 sum axis=1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT8 sum axis=1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT16 sum axis=1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT32 sum axis=1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT64 sum axis=1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: BOOL sum axis=1
RESULT: [2,1]
TYPE: %d
SHAPE: [2]
PASS: FLOAT16 sum axis=0 keepdims
RESULT: [[5,7,9]]
TYPE: %d
SHAPE: [1,3]
PASS: FLOAT32 sum axis=0 keepdims
RESULT: [[5,7,9]]
TYPE: %d
SHAPE: [1,3]
PASS: FLOAT64 sum axis=0 keepdims
RESULT: [[5,7,9]]
TYPE: %d
SHAPE: [1,3]
PASS: INT8 sum axis=0 keepdims
RESULT: [[5,7,9]]
TYPE: %d
SHAPE: [1,3]
PASS: INT16 sum axis=0 keepdims
RESULT: [[5,7,9]]
TYPE: %d
SHAPE: [1,3]
PASS: INT32 sum axis=0 keepdims
RESULT: [[5,7,9]]
TYPE: %d
SHAPE: [1,3]
PASS: INT64 sum axis=0 keepdims
RESULT: [[5,7,9]]
TYPE: %d
SHAPE: [1,3]
PASS: BOOL sum axis=0 keepdims
RESULT: [[1,1,1]]
TYPE: %d
SHAPE: [1,3]
PASS: FLOAT16 sum axis=1 keepdims
RESULT: [[6],[15]]
TYPE: %d
SHAPE: [2,1]
PASS: FLOAT32 sum axis=1 keepdims
RESULT: [[6],[15]]
TYPE: %d
SHAPE: [2,1]
PASS: FLOAT64 sum axis=1 keepdims
RESULT: [[6],[15]]
TYPE: %d
SHAPE: [2,1]
PASS: INT8 sum axis=1 keepdims
RESULT: [[6],[15]]
TYPE: %d
SHAPE: [2,1]
PASS: INT16 sum axis=1 keepdims
RESULT: [[6],[15]]
TYPE: %d
SHAPE: [2,1]
PASS: INT32 sum axis=1 keepdims
RESULT: [[6],[15]]
TYPE: %d
SHAPE: [2,1]
PASS: INT64 sum axis=1 keepdims
RESULT: [[6],[15]]
TYPE: %d
SHAPE: [2,1]
PASS: BOOL sum axis=1 keepdims
RESULT: [[2],[1]]
TYPE: %d
SHAPE: [2,1]
PASS: FLOAT16 sum axis=-1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: FLOAT32 sum axis=-1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: FLOAT64 sum axis=-1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT8 sum axis=-1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT16 sum axis=-1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT32 sum axis=-1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: INT64 sum axis=-1
RESULT: [6,15]
TYPE: %d
SHAPE: [2]
PASS: BOOL sum axis=-1
RESULT: [2,1]
TYPE: %d
SHAPE: [2]
PASS: Error on invalid axis: sum: axis 2 is out of bounds for tensor with 2 dimensions
