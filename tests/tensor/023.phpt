--TEST--
Check Tensor constants
--EXTENSIONS--
ort
--FILE--
<?php
echo "UNDEFINED: " . ONNX\Tensor::UNDEFINED . "\n";
echo "FLOAT: " . ONNX\Tensor::FLOAT . "\n";
echo "DOUBLE: " . ONNX\Tensor::DOUBLE . "\n";
echo "UINT8: " . ONNX\Tensor::UINT8 . "\n";
echo "INT8: " . ONNX\Tensor::INT8 . "\n";
echo "UINT16: " . ONNX\Tensor::UINT16 . "\n";
echo "INT16: " . ONNX\Tensor::INT16 . "\n";
echo "UINT32: " . ONNX\Tensor::UINT32 . "\n";
echo "INT32: " . ONNX\Tensor::INT32 . "\n";
echo "INT64: " . ONNX\Tensor::INT64 . "\n";
echo "BOOL: " . ONNX\Tensor::BOOL;
?>
--EXPECTF--
UNDEFINED: %d
FLOAT: %d
DOUBLE: %d
UINT8: %d
INT8: %d
UINT16: %d
INT16: %d
UINT32: %d
INT32: %d
INT64: %d
BOOL: %d