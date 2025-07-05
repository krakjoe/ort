--TEST--
Check Tensor handles invalid type gracefully
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new \ONNX\Tensor\Persistent(
        "test_tensor",
        [2, 2],
        [[1, 2], [3, 4]],
        42
    );
} catch(\ONNX\Status\Tensor\InvalidType $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
string(%d) "unknown data type (42) provided"