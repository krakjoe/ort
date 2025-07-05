--TEST--
Check Status\Tensor\InvalidShape - slice bounds out of range
--EXTENSIONS--
ort
--FILE--
<?php
try {
    $tensor = new ONNX\Tensor\Persistent(
        "test_tensor",
        [3, 3],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ONNX\Tensor::INT64
    );
    
    $slice = $tensor->getSlice([0, 0], [5, 2]); // 5 > 3 (shape)
} catch(ONNX\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK