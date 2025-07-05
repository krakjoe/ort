--TEST--
Check Status\Tensor\InvalidData - wrong data type
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ONNX\Tensor\Persistent(
        "invalid_tensor",
        [2, 2],
        [[1, 2], [3, "string"]], // string instead of number
        ONNX\Tensor::INT64
    );
} catch(ONNX\Status\Tensor\InvalidData $e) {
    echo "OK";
}
?>
--EXPECT--
OK