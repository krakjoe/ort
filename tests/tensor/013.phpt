--TEST--
Check Status\Tensor\InvalidShape - negative dimension
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ONNX\Tensor\Persistent(
        "invalid_tensor",
        [2, -1],
        [[1, 2], [3, 4]],
        ONNX\Tensor::INT64
    );
} catch(ONNX\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK