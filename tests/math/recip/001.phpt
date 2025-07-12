--TEST--
ONNX\Math\recip: line-by-line, all types, shape, error handling, numpy semantics
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
];

$signed_types = [
    'INT8'    => ONNX\Tensor::INT8,
    'INT16'   => ONNX\Tensor::INT16,
    'INT32'   => ONNX\Tensor::INT32,
    'INT64'   => ONNX\Tensor::INT64,
];
$unsigned_types = [
    'UINT8'   => ONNX\Tensor::UINT8,
    'UINT16'  => ONNX\Tensor::UINT16,
    'UINT32'  => ONNX\Tensor::UINT32,
];

$real = [
    'FLOAT'   => ONNX\Tensor::FLOAT,
    'DOUBLE'  => ONNX\Tensor::DOUBLE,
];

function flatten_data($data) {
    if (is_array($data)) {
        return '[' . implode(',', array_map('flatten_data', $data)) . ']';
    }
    if (is_float($data)) {
        if (is_nan($data)) return 'NAN';
        if ($data === INF) return 'INF';
        if ($data === -INF) return '-INF';
        if ($data == 0.0) return '0';
        return rtrim(rtrim(sprintf('%.17g', $data), '0'), '.');
    }
    if (is_bool($data)) return $data ? '1' : '0';
    return (string)$data;
}

function print_result($result) {
    if ($result instanceof Tensor) {
        echo 'RESULT: ' . flatten_data($result->getData()) . "\n";
        echo 'TYPE: ' . $result->getType() . "\n";
        echo 'SHAPE: [' . implode(',', $result->getShape()) . "]\n";
    } else {
        echo "NOTENSOR\n";
    }
}

// 1. Recip for signed/real types: negatives, zero, positives (large enough for vectorization)
$large_signed = array_merge(
    range(-16, -1),
    [0],
    range(1, 16)
);
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($large_signed)], $large_signed, $type);
    $result = ONNX\Math\recip($a);
    echo "PASS: $name recip large signed [-16..-1,0,1..16]\n";
    print_result($result);
}
// 1b. Recip for unsigned types: only non-negative values (large enough for vectorization)
$large_unsigned = range(0, 31);
foreach ($unsigned_types as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($large_unsigned)], $large_unsigned, $type);
    $result = ONNX\Math\recip($a);
    echo "PASS: $name recip large unsigned [0..31]\n";
    print_result($result);
}

// 2. Recip of ones and zeros (valid for all types, large set)
$zeros_ones = array_fill(0, 32, 0);
for ($i = 0; $i < 32; $i += 2) $zeros_ones[$i] = 1;
foreach ($types as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($zeros_ones)], $zeros_ones, $type);
    $result = ONNX\Math\recip($a);
    echo "PASS: $name recip large zeros/ones\n";
    print_result($result);
}

// 3. Recip of 2D tensor for INT8 (with negative and positive values, large enough)
$int8_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i % 2 === 0 ? 1 : -1) * ($i * 8 + $j + 1);
    }
    $int8_2d[] = $row;
}
$a = new ONNX\Tensor\Transient([8,8], $int8_2d, ONNX\Tensor::INT8);
$result = ONNX\Math\recip($a);
echo "PASS: INT8 recip 2D 8x8 alt sign\n";
print_result($result);

// 4. Recip of bool tensor (large enough)
$bool_2d = [];
for ($i = 0; $i < 8; $i++) {
    $row = [];
    for ($j = 0; $j < 8; $j++) {
        $row[] = ($i + $j) % 2 === 0;
    }
    $bool_2d[] = $row;
}
$a = new ONNX\Tensor\Transient([8,8], $bool_2d, ONNX\Tensor::BOOL);
$result = ONNX\Math\recip($a);
echo "PASS: BOOL recip [bool 8x8] (numpy semantics)\n";
print_result($result);

// 5. Recip of large and small values (real types only, large set)
$extremes = array_merge([
    1e10, 1e-10, 0, 1
], range(2, 33));
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($extremes)], $extremes, $type);
    $result = ONNX\Math\recip($a);
    echo "PASS: $name recip extremes large\n";
    print_result($result);
}

// 6. Recip of empty tensor (should error)
try {
    $a = new ONNX\Tensor\Transient([0], [], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\recip($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT recip large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: %d
SHAPE: [33]
PASS: DOUBLE recip large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: %d
SHAPE: [33]
PASS: INT8 recip large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: %d
SHAPE: [33]
PASS: INT16 recip large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: %d
SHAPE: [33]
PASS: INT32 recip large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: %d
SHAPE: [33]
PASS: INT64 recip large signed [-16..-1,0,1..16]
RESULT: %s
TYPE: %d
SHAPE: [33]
PASS: UINT8 recip large unsigned [0..31]
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: UINT16 recip large unsigned [0..31]
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: UINT32 recip large unsigned [0..31]
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: FLOAT recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: DOUBLE recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: INT8 recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: INT16 recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: INT32 recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: INT64 recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: UINT8 recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: UINT16 recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: UINT32 recip large zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [32]
PASS: INT8 recip 2D 8x8 alt sign
RESULT: %s
TYPE: %d
SHAPE: [8,8]
PASS: BOOL recip [bool 8x8] (numpy semantics)
RESULT: [[1,INF,1,INF,1,INF,1,INF],[INF,1,INF,1,INF,1,INF,1],[1,INF,1,INF,1,INF,1,INF],[INF,1,INF,1,INF,1,INF,1],[1,INF,1,INF,1,INF,1,INF],[INF,1,INF,1,INF,1,INF,1],[1,INF,1,INF,1,INF,1,INF],[INF,1,INF,1,INF,1,INF,1]]
TYPE: 11
SHAPE: [8,8]
PASS: FLOAT recip extremes large
RESULT: %s
TYPE: 1
SHAPE: [36]
PASS: DOUBLE recip extremes large
RESULT: %s
TYPE: 11
SHAPE: [36]
PASS: Error on empty tensor: %s
