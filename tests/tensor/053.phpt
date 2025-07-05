--TEST--
Check Tensor debug info edge cases and comprehensive coverage
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test debug info for different tensor types to ensure comprehensive coverage

// Test INT8 tensor
$int8_tensor = new ONNX\Tensor\Persistent(
    "int8_tensor",
    [2],
    [127, -128],
    ONNX\Tensor::INT8
);
echo "=== INT8 Tensor Debug Info ===\n";
var_dump($int8_tensor);

// Test UINT16 tensor  
$uint16_tensor = new ONNX\Tensor\Persistent(
    "uint16_tensor",
    [3],
    [0, 32767, 65535],
    ONNX\Tensor::UINT16
);
echo "\n=== UINT16 Tensor Debug Info ===\n";
var_dump($uint16_tensor);

// Test DOUBLE tensor
$double_tensor = new ONNX\Tensor\Persistent(
    "double_tensor",
    [1, 2],
    [[3.141592653589793, 2.718281828459045]],
    ONNX\Tensor::DOUBLE
);
echo "\n=== DOUBLE Tensor Debug Info ===\n";
var_dump($double_tensor);

// Test 1D tensor (different shape)
$tensor_1d = new ONNX\Tensor\Persistent(
    "tensor_1d",
    [5],
    [1, 2, 3, 4, 5],
    ONNX\Tensor::INT64
);
echo "\n=== 1D Tensor Debug Info ===\n";
var_dump($tensor_1d);

// Test 4D tensor (complex shape)
$tensor_4d = new ONNX\Tensor\Persistent(
    "tensor_4d",
    [1, 2, 2, 1],
    [[[[1], [2]], [[3], [4]]]],
    ONNX\Tensor::UINT32
);
echo "\n=== 4D Tensor Debug Info ===\n";
var_dump($tensor_4d);
?>
--EXPECTF--
=== INT8 Tensor Debug Info ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(11) "int8_tensor"
  ["shape"]=>
  array(1) {
    [0]=>
    int(2)
  }
}

=== UINT16 Tensor Debug Info ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(13) "uint16_tensor"
  ["shape"]=>
  array(1) {
    [0]=>
    int(3)
  }
}

=== DOUBLE Tensor Debug Info ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(13) "double_tensor"
  ["shape"]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
}

=== 1D Tensor Debug Info ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(9) "tensor_1d"
  ["shape"]=>
  array(1) {
    [0]=>
    int(5)
  }
}

=== 4D Tensor Debug Info ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(9) "tensor_4d"
  ["shape"]=>
  array(4) {
    [0]=>
    int(1)
    [1]=>
    int(2)
    [2]=>
    int(2)
    [3]=>
    int(1)
  }
}
