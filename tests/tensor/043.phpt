--TEST--
Check Tensor getData method with BOOL type
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
    [2, 2],
    [[true, false], [false, true]],
    ORT\Tensor::BOOL
);

$data = $tensor->getData();
var_dump($data);
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    bool(true)
    [1]=>
    bool(false)
  }
  [1]=>
  array(2) {
    [0]=>
    bool(false)
    [1]=>
    bool(true)
  }
}
