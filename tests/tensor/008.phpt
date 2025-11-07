--TEST--
Check Tensor with FLOAT type
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "float_tensor",
    [2, 2],
    [[1.5, 2.5], [3.5, 4.5]],
    ORT\Tensor::FLOAT32
);
echo "Type: " . $tensor->getType() . "\n";
echo "Shape: " . implode('x', $tensor->getShape());
?>
--EXPECTF--
Type: %d
Shape: 2x2