--TEST--
Check Status\Model\InvalidFile no file
--EXTENSIONS--
ort
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