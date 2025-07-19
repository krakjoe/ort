--TEST--
Check Tensor getType method
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[1, 2], [3, 4]],
    ORT\Tensor::INT32
);
echo $tensor->getTypeName();
?>
--EXPECT--
INT32