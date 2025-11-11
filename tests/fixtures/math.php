<?php
use \ORT\Tensor;

function flatten_data($data) {
    if (\is_array($data)) {
        return '[' . \implode(',', 
            \array_map('flatten_data',
                $data)) . ']';
    }

    if (\is_float($data)) {
        if (\is_nan($data)) 
            return 'NAN';
        if ($data === INF)
            return 'INF';
        if ($data === -INF)
            return '-INF';
        if ($data == 0.0)
            return '0';

        // Print floats with max precision
        return \rtrim(\rtrim(\sprintf('%.17g', $data), '0'), '.');
    }

    if (\is_bool($data))
        return $data ? 'true' : 'false';

    if (\is_int($data)) {
        return \sprintf('%d', $data);
    }

    return (string) $data;
}

function print_result($result) {
    if ($result instanceof Tensor) {
        echo 'RESULT: ' . flatten_data($result->getData()) . "\n";
        echo 'TYPE: ' . $result->getTypeName() . "\n";
        echo 'SHAPE: [' . \implode(',', $result->getShape()) . "]\n";
    } else {
        echo "NOTENSOR\n";
    }
}

$types = [
    'FLOAT16' => ORT\Tensor::FLOAT16,
    'FLOAT32' => ORT\Tensor::FLOAT32,
    'FLOAT64' => ORT\Tensor::FLOAT64,
    'INT8'    => ORT\Tensor::INT8,
    'INT16'   => ORT\Tensor::INT16,
    'INT32'   => ORT\Tensor::INT32,
    'INT64'   => ORT\Tensor::INT64,
    'UINT8'   => ORT\Tensor::UINT8,
    'UINT16'  => ORT\Tensor::UINT16,
    'UINT32'  => ORT\Tensor::UINT32,
    'BOOL'    => ORT\Tensor::BOOL,
];

$signed_types = [
    'INT8'    => ORT\Tensor::INT8,
    'INT16'   => ORT\Tensor::INT16,
    'INT32'   => ORT\Tensor::INT32,
    'INT64'   => ORT\Tensor::INT64,
];
$unsigned_types = [
    'UINT8'   => ORT\Tensor::UINT8,
    'UINT16'  => ORT\Tensor::UINT16,
    'UINT32'  => ORT\Tensor::UINT32,
];

$real = [
    'FLOAT16' => ORT\Tensor::FLOAT16,
    'FLOAT32' => ORT\Tensor::FLOAT32,
    'FLOAT64' => ORT\Tensor::FLOAT64,
];

$large_sizes = [
    'FLOAT16' => 8192,
    'FLOAT32' => 4096,
    'FLOAT64' => 4096,
    'INT8'    => 16384,
    'INT16'   => 8192,
    'INT32'   => 4096,
    'INT64'   => 4096,
    'UINT8'   => 16384,
    'UINT16'  => 8192,
    'UINT32'  => 4096
];