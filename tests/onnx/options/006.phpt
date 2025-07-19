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
$options = new \ORT\Options();

$options->setLogVerbosity(
    \ORT\Options::LVERBOSE);
$options->setLogVerbosity(
    \ORT\Options::LINFO);
$options->setLogVerbosity(
    \ORT\Options::LERROR);
$options->setLogVerbosity(
    \ORT\Options::LWARNING);
$options->setLogVerbosity(
    \ORT\Options::LFATAL);

echo "OK";
?>
--EXPECT--
OK