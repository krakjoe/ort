--TEST--
Check Tensor with 4D array
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "4d_tensor",
    [1, 1, 2, 2],
    [[[[1, 2], [3, 4]]]],
    ONNX\Tensor::INT64
);
print_r($tensor->getShape());
?>
--EXPECT--
Array
(
    [0] => 1
    [1] => 1
    [2] => 2
    [3] => 2
)