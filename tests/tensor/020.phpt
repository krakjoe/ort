--TEST--
Check Status\Tensor\InvalidShape - slice start >= end
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
    
    $slice = $tensor->getSlice([2, 1], [2, 3]); // start[0] == end[0]
} catch(ORT\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK
