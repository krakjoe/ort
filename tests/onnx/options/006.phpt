--TEST--
Check Options::setLogVerbosity
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

$options->setLogVerbosity(
    \ONNX\Options::LVERBOSE);
$options->setLogVerbosity(
    \ONNX\Options::LINFO);
$options->setLogVerbosity(
    \ONNX\Options::LERROR);
$options->setLogVerbosity(
    \ONNX\Options::LWARNING);
$options->setLogVerbosity(
    \ONNX\Options::LFATAL);

echo "OK";
?>
--EXPECT--
OK