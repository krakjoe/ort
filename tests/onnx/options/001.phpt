--TEST--
Check Options construct/destruct
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
var_dump(new \ORT\Options());
?>
--EXPECTF--
object(ORT\Options)#%d (0) {
}