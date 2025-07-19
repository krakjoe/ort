--TEST--
Check Tensor getSlice method - basic slice
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [3, 3],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ORT\Tensor::INT64
);

$slice = $tensor->getSlice([0, 0], [2, 2]);
print_r($slice->getShape());
?>
--EXPECT--
Array
(
    [0] => 2
    [1] => 2
)