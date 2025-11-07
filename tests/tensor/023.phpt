--TEST--
Check Tensor constants
--EXTENSIONS--
ort
--FILE--
<?php
echo "UNDEFINED: " . ORT\Tensor::UNDEFINED . "\n";
echo "FLOAT: " . ORT\Tensor::FLOAT32 . "\n";
echo "DOUBLE: " . ORT\Tensor::FLOAT64 . "\n";
echo "UINT8: " . ORT\Tensor::UINT8 . "\n";
echo "INT8: " . ORT\Tensor::INT8 . "\n";
echo "UINT16: " . ORT\Tensor::UINT16 . "\n";
echo "INT16: " . ORT\Tensor::INT16 . "\n";
echo "UINT32: " . ORT\Tensor::UINT32 . "\n";
echo "INT32: " . ORT\Tensor::INT32 . "\n";
echo "INT64: " . ORT\Tensor::INT64 . "\n";
echo "BOOL: " . ORT\Tensor::BOOL;
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