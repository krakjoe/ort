--TEST--
Check Options construct/destruct
--EXTENSIONS--
ort
--FILE--
<?php
var_dump(new \ONNX\Options());
?>
--EXPECTF--
object(ONNX\Options)#%d (0) {
}