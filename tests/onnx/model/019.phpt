--TEST--
Check Model serialization is denied
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
$model = new ORT\Model("mismatch_test",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));

try {
    $serialized = serialize($model);
    echo "Serialization should not be allowed\n";
} catch (Exception $e) {
    echo "Serialization correctly denied\n";
}
?>
--EXPECT--
Serialization correctly denied