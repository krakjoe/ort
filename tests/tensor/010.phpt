--TEST--
Check Tensor with BOOL type
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "bool_tensor",
    [2, 2],
    [[true, false], [false, true]],
    ORT\Tensor::BOOL
);
echo "Type: " . $tensor->getTypeName();
?>
--EXPECT--
Type: BOOL