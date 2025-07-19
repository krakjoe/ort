--TEST--
Check Tensor getShape method
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [2, 3, 4],
    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 
     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]],
    ORT\Tensor::INT64
);
print_r($tensor->getShape());
?>
--EXPECT--
Array
(
    [0] => 2
    [1] => 3
    [2] => 4
)