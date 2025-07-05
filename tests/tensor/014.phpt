--TEST--
Check Status\Tensor\InvalidShape - non-packed array
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ONNX\Tensor\Persistent(
        "invalid_tensor",
        [0 => 2, 2 => 3], // missing index 1
        [[1, 2, 3], [4, 5, 6]],
        ONNX\Tensor::INT64
    );
} catch(ONNX\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK