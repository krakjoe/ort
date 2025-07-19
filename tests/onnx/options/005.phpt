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
$options = new \ORT\Options();

$options->setLogSeverity(
    \ORT\Options::LINFO);
$options->setLogSeverity(
    \ORT\Options::LERROR);
$options->setLogSeverity(
    \ORT\Options::LWARNING);
$options->setLogSeverity(
    \ORT\Options::LFATAL);

echo "OK";
?>
--EXPECT--
OK