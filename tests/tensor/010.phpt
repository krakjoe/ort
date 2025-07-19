--TEST--
Check Tensor with BOOL type
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "bool_tensor",
    [2, 2],
    [[true, false], [false, true]],
    ONNX\Tensor::BOOL
);
echo "Type: " . $tensor->getTypeName();
?>
--EXPECT--
Type: BOOL