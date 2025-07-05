--TEST--
Check Options::setExecutionMode
--EXTENSIONS--
ort
--FILE--
<?php
$options = new \ONNX\Options();

$options->setExecutionMode(
    \ONNX\Options::ESEQUENTIAL);

var_dump($options);

$options->setExecutionMode(
    \ONNX\Options::EPARALLEL);

var_dump($options);

try {
    $options->setExecutionMode(42);
} catch(\ONNX\Status\Error $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
object(ONNX\Options)#%d (0) {
}
object(ONNX\Options)#%d (0) {
}
string(%d) "failed to set execution mode: execution_mode is not valid"
