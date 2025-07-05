--TEST--
Check Tensor getType method
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[1, 2], [3, 4]],
    ONNX\Tensor::INT32
);
echo $tensor->getType();
?>
--EXPECT--
6