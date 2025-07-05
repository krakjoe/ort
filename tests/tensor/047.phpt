--TEST--
Check Tensor getData method with invalid offset parameter
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

// Test negative offset
try {
    $data = $tensor->getData(-1);
} catch (ONNX\Status\Tensor\InvalidData $e) {
    echo "Caught negative offset: " . $e->getMessage() . "\n";
}

// Test offset equal to elements count
$data = $tensor->getData(6); // Should return empty array (6 elements total)
echo "Empty array for offset at boundary: ";
var_dump($data);

// Test offset greater than elements count
try {
    $data = $tensor->getData(10);
} catch (ONNX\Status\Tensor\InvalidData $e) {
    echo "Caught large offset: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
Caught negative offset: offset -1 out of range [0, 6]
Empty array for offset at boundary: array(0) {
}
Caught large offset: offset 10 out of range [0, 6]
