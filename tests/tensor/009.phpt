--TEST--
Check Tensor with DOUBLE type
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "double_tensor",
    [1, 3],
    [[1.123456789, 2.987654321, 3.141592653]],
    ONNX\Tensor::DOUBLE
);
echo "Type: " . $tensor->getType();
?>
--EXPECT--
Type: 11