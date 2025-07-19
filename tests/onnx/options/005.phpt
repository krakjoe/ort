--TEST--
Check Options::setLogSeverity
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
$options = new \ONNX\Options();

$options->setLogSeverity(
    \ONNX\Options::LINFO);
$options->setLogSeverity(
    \ONNX\Options::LERROR);
$options->setLogSeverity(
    \ONNX\Options::LWARNING);
$options->setLogSeverity(
    \ONNX\Options::LFATAL);

echo "OK";
?>
--EXPECT--
OK