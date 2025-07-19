--TEST--
Check Model iterators
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
$model = new ORT\Model("test_model",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));

printf("Model: %s\n", $model->getName());
printf("Input(%d):\n", $model->getInputCount());
foreach ($model->getInputIterator() as $idx => $input) {
    printf("\t%d --> %s\n", $idx, $input);
}

printf("Output(%d):\n", $model->getOutputCount());
foreach ($model->getOutputIterator() as $idx => $input) {
    printf("\t%d --> %s\n", $idx, $input);
}
?>
--EXPECTF--
Model: test_model
Input(2):
%w0 --> A
%w1 --> B
Output(1):
%w0 --> C
