--TEST--
Check Options::setLogId
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

$options->setLogId("\0");
$options->setLogId("");
$options->setLogId("php-ort");

echo "OK";
?>
--EXPECT--
OK