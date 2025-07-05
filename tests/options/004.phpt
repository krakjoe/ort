--TEST--
Check Options::setLogId
--EXTENSIONS--
ort
--FILE--
<?php
$options = new \ONNX\Options();

$options->setLogId("\0");
$options->setLogId("");
$options->setLogId("php-ort");

echo "OK";
?>
--EXPECT--
OK