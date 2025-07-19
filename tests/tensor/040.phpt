--TEST--
Check Status\Tensor\InvalidShape for getData with depth out of range
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
try {
    $tensor = new ORT\Tensor\Persistent(
        "test_tensor",
        [2, 3],
        [[1, 2, 3], [4, 5, 6]],
        ORT\Tensor::INT32
    );
    
    $tensor->getData(0, 2); // Max valid depth is 1 (2 dimensions total)
} catch (ORT\Status\Tensor\InvalidShape $e) {
    echo "Exception caught: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
Exception caught: depth 2 out of range [0, 2)
