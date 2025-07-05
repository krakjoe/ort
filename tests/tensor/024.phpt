--TEST--
Check Tensor with mixed numeric types for FLOAT
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "mixed_tensor",
    [2, 2],
    [[1, 2.5], [3.7, 4]], // mix of int and float
    ONNX\Tensor::FLOAT
);
echo "OK";
?>
--EXPECT--
OK