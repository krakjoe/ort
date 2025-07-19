--TEST--
Check Options::setGraphOptimization
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
$options = new \ORT\Options();

$options->setGraphOptimization(
    \ORT\Options::ONONE);

var_dump($options);

$options->setGraphOptimization(
    \ORT\Options::OBASIC);

var_dump($options);

$options->setGraphOptimization(
    \ORT\Options::OEXTENDED);

var_dump($options);

$options->setGraphOptimization(
    \ORT\Options::OALL);

var_dump($options);

try {
    $options->setGraphOptimization(42);
} catch (\ORT\Status\Error $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
object(ORT\Options)#%d (0) {
}
object(ORT\Options)#%d (0) {
}
object(ORT\Options)#%d (0) {
}
object(ORT\Options)#%d (0) {
}
string(%d) "failed to set graph optimization: graph_optimization_level is not valid"