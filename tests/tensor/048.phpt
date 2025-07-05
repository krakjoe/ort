--TEST--
Check Tensor::getData depth
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "test_tensor",
    [2, 3],
    [[1, 2, 3], [4, 5, 6]],
    ONNX\Tensor::INT32
);

// Test negative depth
try {
    $data = $tensor->getData(0, -1);
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "Caught negative depth: " . $e->getMessage() . "\n";
}

// Test depth equal to dimensions count
try {
    $data = $tensor->getData(0, 2); // tensor has 2 dimensions, so 2 is out of range
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "Caught depth at boundary: " . $e->getMessage() . "\n";
}

// Test depth greater than dimensions count
try {
    $data = $tensor->getData(0, 5);
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "Caught large depth: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
Caught negative depth: depth -1 out of range [0, 2)
Caught depth at boundary: depth 2 out of range [0, 2)
Caught large depth: depth 5 out of range [0, 2)
