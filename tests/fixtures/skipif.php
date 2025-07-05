<?php
if (!\file_exists(\sprintf(
    "%s/add.onnx",
    dirname(__FILE__)
))) {
    die ("skip: fixtures/add.onnx could not be found");
}

if (!\file_exists(\sprintf(
    "%s/sub.onnx",
    dirname(__FILE__)
))) {
    die ("skip: fixtures/sub.onnx could not be found");
}
?>