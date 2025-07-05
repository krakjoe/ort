--TEST--
Check Status\Tensor\NotFound
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ONNX\Tensor\Persistent("not.found");
} catch(ONNX\Status\Tensor\NotFound) {
    echo "OK";
}
?>
--EXPECT--
OK