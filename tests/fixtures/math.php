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
    'FLOAT'   => ORT\Tensor::FLOAT,
    'DOUBLE'  => ORT\Tensor::DOUBLE,
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
    'FLOAT'   => ORT\Tensor::FLOAT,
    'DOUBLE'  => ORT\Tensor::DOUBLE,
];