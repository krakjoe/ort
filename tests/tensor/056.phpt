--TEST--
Comprehensive Tensor debug info coverage test
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test comprehensive debug info coverage including edge cases

echo "=== Test 1: Uninitialized tensor (NULL object) ===\n";
$reflectionClass = new ReflectionClass('ONNX\\Tensor\\Persistent');
$uninitializedTensor = $reflectionClass->newInstanceWithoutConstructor();
var_dump($uninitializedTensor);

echo "\n=== Test 2: Tensor with name ===\n";
$tensorWithName = new ONNX\Tensor\Persistent(
    "named_tensor",
    [2, 3],
    [[1, 2, 3], [4, 5, 6]],
    ONNX\Tensor::INT32
);
var_dump($tensorWithName);

echo "\n=== Test 3: Tensor without name (just lookup) ===\n";
// First create a tensor with name
$lookup = new ONNX\Tensor\Persistent(
    "lookup_test",
    [3],
    [10, 20, 30],
    ONNX\Tensor::INT64
);
// Now try to find it (this should reuse the same tensor)
$lookupTensor = new ONNX\Tensor\Persistent("lookup_test");
var_dump($lookupTensor);

echo "\n=== Test 4: Different tensor types in debug info ===\n";
$boolTensor = new ONNX\Tensor\Persistent(
    "bool_debug",
    [2, 2],
    [[true, false], [false, true]],
    ONNX\Tensor::BOOL
);
var_dump($boolTensor);

?>
--EXPECTF--
=== Test 1: Uninitialized tensor (NULL object) ===
object(ONNX\Tensor\Persistent)#%d (%d) {
}

=== Test 2: Tensor with name ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(12) "named_tensor"
  ["shape"]=>
  array(2) {
    [0]=>
    int(2)
    [1]=>
    int(3)
  }
}

=== Test 3: Tensor without name (just lookup) ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(11) "lookup_test"
  ["shape"]=>
  array(1) {
    [0]=>
    int(3)
  }
}

=== Test 4: Different tensor types in debug info ===
object(ONNX\Tensor\Persistent)#%d (%d) {
  ["persistent"]=>
  bool(true)
  ["type"]=>
  int(%d)
  ["name"]=>
  string(10) "bool_debug"
  ["shape"]=>
  array(2) {
    [0]=>
    int(2)
    [1]=>
    int(2)
  }
}
