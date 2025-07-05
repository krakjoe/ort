--TEST--
Check Status\Tensor\InvalidData for getData with negative offset
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
    
    $tensor->getData(-1);
} catch (ONNX\Status\Tensor\InvalidData $e) {
    echo "Exception caught: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
Exception caught: offset -1 out of range [0, 6]
