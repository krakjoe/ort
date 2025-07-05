--TEST--
Check Tensor getSlice with negative indices
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

$slice = $tensor->getSlice([-2, -2], [-1, -1]); // last 1x1 element
print_r($slice->getShape());
?>
--EXPECT--
Array
(
    [0] => 1
    [1] => 1
)