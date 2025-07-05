--TEST--
Check Options defaults
--EXTENSIONS--
ort
--SKIPIF--
<?php
if (!file_exists(\sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)))) {
    die("skip add model cannot be found");
}
?>
--FILE--
<?php
$model = new \ONNX\Model("test_model",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));
$runtime = 
    new \ONNX\Runtime($model);
echo "OK";
?>
--EXPECT--
OK