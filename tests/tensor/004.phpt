--TEST--
Check Status\Tensor\InvalidData no valid data found
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ORT\Tensor\Persistent("my.tensor", [2, 3], []);
} catch(ORT\Status\Tensor\InvalidData $status) {
    var_dump($status->getMessage());
}
?>
--EXPECT--
string(61) "validation of data according to the shape provided has failed"