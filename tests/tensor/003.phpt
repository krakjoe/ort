--TEST--
Check Tensor getName method
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[1, 2], [3, 4]],
    ONNX\Tensor::INT64
);
echo $tensor->getName();
?>
--EXPECT--
test_tensor