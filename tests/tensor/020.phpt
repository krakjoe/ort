--TEST--
Check Status\Tensor\InvalidShape - slice start >= end
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
    
    $slice = $tensor->getSlice([2, 1], [2, 3]); // start[0] == end[0]
} catch(ONNX\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK
