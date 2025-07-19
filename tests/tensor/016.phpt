--TEST--
Check Status\Tensor\InvalidData - wrong data type
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ORT\Tensor\Persistent(
        "invalid_tensor",
        [2, 2],
        [[1, 2], [3, "string"]], // string instead of number
        ORT\Tensor::INT64
    );
} catch(ORT\Status\Tensor\InvalidData $e) {
    echo "OK";
}
?>
--EXPECT--
OK