--TEST--
Check Tensor getData method with no parameters
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [2, 3],
    [[1, 2, 3], [4, 5, 6]],
    ORT\Tensor::INT32
);

$data = $tensor->getData();
var_dump(is_array($data));
var_dump(count($data));
var_dump($data);
?>
--EXPECT--
bool(true)
int(2)
array(2) {
  [0]=>
  array(3) {
    [0]=>
    int(1)
    [1]=>
    int(2)
    [2]=>
    int(3)
  }
  [1]=>
  array(3) {
    [0]=>
    int(4)
    [1]=>
    int(5)
    [2]=>
    int(6)
  }
}
