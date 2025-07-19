--TEST--
Check Status\Tensor\InvalidData - mismatched data structure
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ORT\Tensor\Persistent(
        "invalid_tensor",
        [2, 2],
        [[1, 2], [3]], // second row has wrong length
        ORT\Tensor::INT64
    );
} catch(ORT\Status\Tensor\InvalidData $e) {
    echo "OK";
}
?>
--EXPECT--
OK