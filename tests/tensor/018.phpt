--TEST--
Check Tensor getSlice method - with axis parameter
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "test_tensor",
    [3, 3],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ONNX\Tensor::INT64
);

$slice = $tensor->getSlice([1], [3], [0]); // slice first dimension from 1 to 3
print_r($slice->getShape());
?>
--EXPECT--
Array
(
    [0] => 2
    [1] => 3
)