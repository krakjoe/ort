--TEST--
Check Tensor with 1D array
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "1d_tensor",
    [5],
    [1, 2, 3, 4, 5],
    ONNX\Tensor::INT64
);
print_r($tensor->getShape());
?>
--EXPECT--
Array
(
    [0] => 5
)