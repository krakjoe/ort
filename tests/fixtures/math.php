<?php
use \ONNX\Tensor;

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