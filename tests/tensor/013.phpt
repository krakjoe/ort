--TEST--
Check Status\Tensor\InvalidShape - negative dimension
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ORT\Tensor\Persistent(
        "invalid_tensor",
        [2, -1],
        [[1, 2], [3, 4]],
        ORT\Tensor::INT64
    );
} catch(ORT\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK