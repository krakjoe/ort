--TEST--
Check Status\Tensor\InvalidShape for getData with negative depth
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
    
    $tensor->getData(0, -1);
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "Exception caught: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
Exception caught: depth -1 out of range [0, 2)
