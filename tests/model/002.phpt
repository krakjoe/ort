--TEST--
Check Status\Model\InvalidFile invalid path
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ONNX\Model("invalid.model", "/nonexistent/path/model.onnx");
} catch(ONNX\Status\Model\InvalidFile $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
string(%d) "model invalid.model could not be found, /nonexistent/path/model.onnx is not a valid path"
