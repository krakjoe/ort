--TEST--
ONNX\Tensor::getTypeName()
--EXTENSIONS--
ort
--FILE--
<?php
foreach ([
    \ONNX\Tensor::DOUBLE,
    \ONNX\Tensor::FLOAT,
    \ONNX\Tensor::INT8,
    \ONNX\Tensor::INT16,
    \ONNX\Tensor::INT32,
    \ONNX\Tensor::INT64,
    \ONNX\Tensor::UINT8,
    \ONNX\Tensor::UINT16,
    \ONNX\Tensor::UINT32,
    \ONNX\Tensor::BOOL
] as $type) {
    $tensor = \ONNX\Tensor\Transient::from(
        $type == \ONNX\Tensor::BOOL ?
            [true, false, true] : [1,2,3], $type);

    if ($type != $tensor->getType()) {
        echo "FAIL: getType returned unexpected things\n";
        exit;
    }

    printf("OK: %s\n", $tensor->getTypeName());
}
?>
--EXPECTF--
OK: DOUBLE
OK: FLOAT
OK: INT8
OK: INT16
OK: INT32
OK: INT64
OK: UINT8
OK: UINT16
OK: UINT32
OK: BOOL