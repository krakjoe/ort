<?php
$types = [
    'int8'   => ONNX\Tensor::INT8,
    'int16'  => ONNX\Tensor::INT16,
    'int32'  => ONNX\Tensor::INT32,
    'int64'  => ONNX\Tensor::INT64,
    'uint8'  => ONNX\Tensor::UINT8,
    'uint16' => ONNX\Tensor::UINT16,
    'uint32' => ONNX\Tensor::UINT32,
    'float32'=> ONNX\Tensor::FLOAT,
    'float64'=> ONNX\Tensor::DOUBLE,
];

$values = [-3, -2, -1, 0, 1, 2, 3];

$funcs = [
    'floor' => 'floor',
    /*
    'round' => 'round',
    'trunc' => 'trunc',
    */
];

foreach ($types as $name => $type) {
    $tensor = new ONNX\Tensor\Transient([count($values)], $values, $type);
    echo "\nType: $name\n";
    foreach ($funcs as $label => $func) {
        $result = call_user_func_array(
            "ONNX\Math\\$func",
            [$tensor]
        );
        printf("%s: type=%s, values=%s\n",
            $func,
            $result->getTypeName(),
            json_encode($result->getData())
        );
    }
}
