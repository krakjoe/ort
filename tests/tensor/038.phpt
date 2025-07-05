--TEST--
Check Tensor getData method with offset equal to elements count returns empty array
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
try {
    $tensor = new ONNX\Tensor\Persistent(
        "test_tensor",
        [2, 3],
        [[1, 2, 3], [4, 5, 6]],
        ONNX\Tensor::INT32
    );
    
    // Offset equal to elements count should return empty array (not throw exception)
    $data = $tensor->getData(6); // 6 elements total, offset 6 should return empty
    var_dump($data);
} catch (ONNX\Status\Tensor\InvalidData $e) {
    echo "Exception caught: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
array(0) {
}
