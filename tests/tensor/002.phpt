--TEST--
Check Tensor construction with valid data
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [2, 3],
    [[1, 2, 3], [4, 5, 6]],
    ORT\Tensor::INT64
);
echo "OK";
?>
--EXPECT--
OK