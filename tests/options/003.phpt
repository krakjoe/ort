--TEST--
Check Options::setGraphOptimization
--EXTENSIONS--
ort
--FILE--
<?php
$options = new \ONNX\Options();

$options->setGraphOptimization(
    \ONNX\Options::ONONE);

var_dump($options);

$options->setGraphOptimization(
    \ONNX\Options::OBASIC);

var_dump($options);

$options->setGraphOptimization(
    \ONNX\Options::OEXTENDED);

var_dump($options);

$options->setGraphOptimization(
    \ONNX\Options::OALL);

var_dump($options);

try {
    $options->setGraphOptimization(42);
} catch (\ONNX\Status\Error $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
object(ONNX\Options)#%d (0) {
}
object(ONNX\Options)#%d (0) {
}
object(ONNX\Options)#%d (0) {
}
object(ONNX\Options)#%d (0) {
}
string(%d) "failed to set graph optimization: graph_optimization_level is not valid"