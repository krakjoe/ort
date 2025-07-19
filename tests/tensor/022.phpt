--TEST--
Check Status\Tensor\InvalidShape - mismatched start/end array lengths
--EXTENSIONS--
ort
--FILE--
<?php
try {
    $tensor = new ORT\Tensor\Persistent(
        "test_tensor",
        [3, 3],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ORT\Tensor::INT64
    );
    
    $slice = $tensor->getSlice([0], [2, 2]); // different lengths
} catch(ORT\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK