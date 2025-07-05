--TEST--
Check Status\Tensor\InvalidData - mismatched data structure
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ONNX\Tensor\Persistent(
        "invalid_tensor",
        [2, 2],
        [[1, 2], [3]], // second row has wrong length
        ONNX\Tensor::INT64
    );
} catch(ONNX\Status\Tensor\InvalidData $e) {
    echo "OK";
}
?>
--EXPECT--
OK