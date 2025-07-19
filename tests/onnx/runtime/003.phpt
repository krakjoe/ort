--TEST--
Check Runtime::run success
--EXTENSIONS--
ort
--SKIPIF--
<?php
include(sprintf(
    "%s/../fixtures/skipif.php",
    dirname(__FILE__)));
?>
--FILE--
<?php
$model = new ORT\Model("test_model",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));

$runtime = new \ORT\Runtime($model);

$A = new ORT\Tensor\Persistent(
    "A",
    [3, 2, 1],
    [
        [[1.0], [2.0]],
        [[3.0], [4.0]],
        [[5.0], [6.0]],
    ],
    ORT\Tensor::FLOAT
);

$B = new \ORT\Tensor\Persistent(
    "B",
    [3, 2, 1],
    [
        [[1.0], [2.0]],
        [[3.0], [4.0]],
        [[5.0], [6.0]],
    ],
    ORT\Tensor::FLOAT
);

$result = $runtime->run([
    "A" => $A,
    "B" => $B
]);

var_dump($result["C"]->getData());
?>
--EXPECT--
array(3) {
  [0]=>
  array(2) {
    [0]=>
    array(1) {
      [0]=>
      float(2)
    }
    [1]=>
    array(1) {
      [0]=>
      float(4)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(1) {
      [0]=>
      float(6)
    }
    [1]=>
    array(1) {
      [0]=>
      float(8)
    }
  }
  [2]=>
  array(2) {
    [0]=>
    array(1) {
      [0]=>
      float(10)
    }
    [1]=>
    array(1) {
      [0]=>
      float(12)
    }
  }
}