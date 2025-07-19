<?php
/* check for ONNX Runtime support */
if (!\class_exists('ORT\\Runtime')) {
    die("skip: ONNX Runtime support is not available\n");
}

/* include main suite skipif file */
include(sprintf(
    "%s/../../fixtures/skipif.php",
    dirname(__FILE__)
));

/* check for model fixtures */
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