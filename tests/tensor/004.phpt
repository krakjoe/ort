--TEST--
Check Status\Tensor\InvalidData no valid data found
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ONNX\Tensor\Persistent("my.tensor", [2, 3], []);
} catch(ONNX\Status\Tensor\InvalidData $status) {
    var_dump($status->getMessage());
}
?>
--EXPECT--
string(61) "validation of data according to the shape provided has failed"