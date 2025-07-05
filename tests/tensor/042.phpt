--TEST--
Check Tensor getData method with DOUBLE type
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[1.5, 2.5], [3.5, 4.5]],
    ONNX\Tensor::DOUBLE
);

$data = $tensor->getData();
var_dump($data);
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1.5)
    [1]=>
    float(2.5)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3.5)
    [1]=>
    float(4.5)
  }
}
