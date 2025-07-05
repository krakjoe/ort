--TEST--
Debug broadcasting test
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Debug Broadcasting Test ===\n";

try {
    $tensor_a = new ONNX\Tensor\Persistent("a", [2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $tensor_b = new ONNX\Tensor\Persistent("b", [3, 2], [[1, 1], [2, 2], [3, 3]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($tensor_a, $tensor_b);
    echo "UNEXPECTED: Add succeeded\n";
} catch (ONNX\Status\Math\InvalidShape $e) {
    echo "EXPECTED: InvalidShape exception caught\n";
} catch (Error $e) {
    echo "UNEXPECTED: Other exception: " . get_class($e) . " - " . $e->getMessage() . "\n";
}

echo "Test completed\n";
?>
--EXPECT--
=== Debug Broadcasting Test ===
EXPECTED: InvalidShape exception caught
Test completed
