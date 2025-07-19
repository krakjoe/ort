--TEST--
Check Tensor isPersistent method
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[1, 2], [3, 4]],
    ORT\Tensor::INT64
);
echo $tensor->isPersistent() ? "true" : "false";
?>
--EXPECT--
true