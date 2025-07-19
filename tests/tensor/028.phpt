--TEST--
Check Tensor reuse - persistent storage
--EXTENSIONS--
ort
--FILE--
<?php
// Create first tensor
$tensor1 = new ORT\Tensor\Persistent(
    "shared_tensor",
    [2, 2],
    [[1, 2], [3, 4]],
    ORT\Tensor::INT64
);

// Create second tensor with same name (should reuse)
$tensor2 = new ORT\Tensor\Persistent("shared_tensor");

echo "Same name: " . ($tensor1->getName() === $tensor2->getName() ? "true" : "false") . "\n";
echo "Same type: " . ($tensor1->getType() === $tensor2->getType() ? "true" : "false") . "\n";
echo "Same shape: " . (serialize($tensor1->getShape()) === serialize($tensor2->getShape()) ? "true" : "false");
?>
--EXPECT--
Same name: true
Same type: true
Same shape: true