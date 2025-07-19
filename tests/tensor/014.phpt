--TEST--
Check Status\Tensor\InvalidShape - non-packed array
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ORT\Tensor\Persistent(
        "invalid_tensor",
        [0 => 2, 2 => 3], // missing index 1
        [[1, 2, 3], [4, 5, 6]],
        ORT\Tensor::INT64
    );
} catch(ORT\Status\Tensor\InvalidShape $e) {
    echo "OK";
}
?>
--EXPECT--
OK