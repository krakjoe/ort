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
string(63) "ragged array: sub-array at dimension 0 has length 0, expected 2"