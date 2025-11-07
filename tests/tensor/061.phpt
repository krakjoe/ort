--TEST--
ORT\Tensor::getTypeName()
--EXTENSIONS--
ort
--FILE--
<?php
foreach ([
    \ORT\Tensor::FLOAT64,
    \ORT\Tensor::FLOAT32,
    \ORT\Tensor::INT8,
    \ORT\Tensor::INT16,
    \ORT\Tensor::INT32,
    \ORT\Tensor::INT64,
    \ORT\Tensor::UINT8,
    \ORT\Tensor::UINT16,
    \ORT\Tensor::UINT32,
    \ORT\Tensor::BOOL
] as $type) {
    $tensor = \ORT\Tensor\Transient::from(
        $type == \ORT\Tensor::BOOL ?
            [true, false, true] : [1,2,3], $type);

    if ($type != $tensor->getType()) {
        echo "FAIL: getType returned unexpected things\n";
        exit;
    }

    printf("OK: %s\n", $tensor->getTypeName());
}
?>
--EXPECTF--
OK: FLOAT64
OK: FLOAT32
OK: INT8
OK: INT16
OK: INT32
OK: INT64
OK: UINT8
OK: UINT16
OK: UINT32
OK: BOOL