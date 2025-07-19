--TEST--
Check Options::setExecutionMode
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

$options->setExecutionMode(
    \ORT\Options::ESEQUENTIAL);

var_dump($options);

$options->setExecutionMode(
    \ORT\Options::EPARALLEL);

var_dump($options);

try {
    $options->setExecutionMode(42);
} catch(\ORT\Status\Error $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
object(ORT\Options)#%d (0) {
}
object(ORT\Options)#%d (0) {
}
string(%d) "failed to set execution mode: execution_mode is not valid"
