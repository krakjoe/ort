--TEST--
Check Tensor getName method
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[1, 2], [3, 4]],
    ORT\Tensor::INT64
);
echo $tensor->getName();
?>
--EXPECT--
test_tensor