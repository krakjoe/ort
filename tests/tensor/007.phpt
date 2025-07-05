--TEST--
Check Tensor isPersistent method
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
echo $tensor->isPersistent() ? "true" : "false";
?>
--EXPECT--
true