--TEST--
Check Status\Model\InvalidFile no file
--EXTENSIONS--
ort
--SKIPIF--
<?php
include(sprintf(
    "%s/../fixtures/skipif.php",
    dirname(__FILE__)));
?>
--FILE--
<?php
try {
    new ONNX\Model("invalid.file");
} catch(ONNX\Status\Model\InvalidFile $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECT--
string(59) "model invalid.file could not be loaded and no file provided"