--TEST--
Check Status\Tensor\NotFound
--EXTENSIONS--
ort
--FILE--
<?php
try {
    new ORT\Tensor\Persistent("not.found");
} catch(ORT\Status\Tensor\NotFound) {
    echo "OK";
}
?>
--EXPECT--
OK