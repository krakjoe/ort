--TEST--
ORT\Tensor\Generator\Random
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

$types = [
    "INT8" => Tensor::INT8,
    "INT16" => Tensor::INT16,
    "INT32" => Tensor::INT32,
    "INT64" => Tensor::INT64,
    "UINT8" => Tensor::UINT8,
    "UINT16" => Tensor::UINT16,
    "UINT32" => Tensor::UINT32,
    "FLOAT32" => Tensor::FLOAT32,
    "FLOAT64" => Tensor::FLOAT64,
];

foreach ($types as $name => $type) {
    $tensor = new Tensor\Transient(
        [6, 6],
        new Tensor\Generator\Random($type),
        $type);
    echo "PASS: $name random tensor 0 params\n";
}

foreach ($types as $name => $type) {
    if ($type == Tensor::FLOAT64 ||
        $type == Tensor::FLOAT32) {
        $param = 42.0;
    } else {
        $param = 42;
    }

    $tensor = new Tensor\Transient(
        [6, 6],
        new Tensor\Generator\Random(
            $type, $param),
        $type);
    echo "PASS: $name random tensor 1 params\n";
}

$params = [
    Tensor::INT8 => [1, 127],
    Tensor::INT16 => [1, 32767],
    Tensor::INT32 => [1, 2147483647],
    Tensor::INT64 => [1, 9223372036854775807],
    Tensor::UINT8 => [0, 255],
    Tensor::UINT16 => [0, 65535],
    Tensor::UINT32 => [0, 4294967295],
    Tensor::FLOAT32 => [0.0, 1.0],
    Tensor::FLOAT64 => [0.0, 1.0],
];

foreach ($types as $name => $type) {
    $tensor = new Tensor\Transient(
        [6, 6],
        new Tensor\Generator\Random(
            $type, ...$params[$type]),
        $type);
    echo "PASS: $name random tensor 2 params\n";
}
?>
--EXPECT--
PASS: INT8 random tensor 0 params
PASS: INT16 random tensor 0 params
PASS: INT32 random tensor 0 params
PASS: INT64 random tensor 0 params
PASS: UINT8 random tensor 0 params
PASS: UINT16 random tensor 0 params
PASS: UINT32 random tensor 0 params
PASS: FLOAT32 random tensor 0 params
PASS: FLOAT64 random tensor 0 params
PASS: INT8 random tensor 1 params
PASS: INT16 random tensor 1 params
PASS: INT32 random tensor 1 params
PASS: INT64 random tensor 1 params
PASS: UINT8 random tensor 1 params
PASS: UINT16 random tensor 1 params
PASS: UINT32 random tensor 1 params
PASS: FLOAT32 random tensor 1 params
PASS: FLOAT64 random tensor 1 params
PASS: INT8 random tensor 2 params
PASS: INT16 random tensor 2 params
PASS: INT32 random tensor 2 params
PASS: INT64 random tensor 2 params
PASS: UINT8 random tensor 2 params
PASS: UINT16 random tensor 2 params
PASS: UINT32 random tensor 2 params
PASS: FLOAT32 random tensor 2 params
PASS: FLOAT64 random tensor 2 params

