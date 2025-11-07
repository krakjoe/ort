--TEST--
Check Tensor debug info (var_dump output)
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test debug info for a standard tensor
$tensor = new ORT\Tensor\Persistent(
    "debug_test_tensor",
    [2, 3],
    [[1, 2, 3], [4, 5, 6]],
    ORT\Tensor::INT32
);

echo "=== Standard Tensor Debug Info ===\n";
var_dump($tensor);

// Test debug info for a tensor with different type and shape
$float_tensor = new ORT\Tensor\Persistent(
    "float_debug_tensor",
    [1, 4],
    [[1.1, 2.2, 3.3, 4.4]],
    ORT\Tensor::FLOAT32
);

echo "\n=== Float Tensor Debug Info ===\n";
var_dump($float_tensor);

// Test debug info for a boolean tensor
$bool_tensor = new ORT\Tensor\Persistent(
    "bool_debug_tensor",
    [3],
    [true, false, true],
    ORT\Tensor::BOOL
);

echo "\n=== Boolean Tensor Debug Info ===\n";
var_dump($bool_tensor);
?>
--EXPECTF--
=== Standard Tensor Debug Info ===
object(ORT\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(17) "debug_test_tensor"
  ["shape"]=>
  array(2) {
    [0]=>
    int(2)
    [1]=>
    int(3)
  }
}

=== Float Tensor Debug Info ===
object(ORT\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(18) "float_debug_tensor"
  ["shape"]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(4)
  }
}

=== Boolean Tensor Debug Info ===
object(ORT\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(17) "bool_debug_tensor"
  ["shape"]=>
  array(1) {
    [0]=>
    int(3)
  }
}
