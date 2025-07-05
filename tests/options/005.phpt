--TEST--
Check Options::setLogSeverity
--EXTENSIONS--
ort
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