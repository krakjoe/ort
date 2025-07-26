--TEST--
ORT\Tensor\Transient::from() named constructor: success, error, and edge cases
--EXTENSIONS--
ort
--FILE--
<?php

function show_result($label, $cb) {
    echo "== $label ==\n";
    try {
        $result = $cb();
        if (is_object($result)) {
            echo "OK: ", get_class($result), "\n";
            if (method_exists($result, 'getShape')) {
                var_dump($result->getShape());
            }
        } else {
            var_dump($result);
        }
    } catch (Throwable $e) {
        echo "EX: ", get_class($e), ": ", $e->getMessage(), "\n";
    }
}

// 1. Success: 2x2 int64 tensor
show_result("success 2x2", function() {
    return ORT\Tensor\Transient::from([[1,2],[3,4]], ORT\Tensor::INT64);
});

// 2. Ragged array (should fail)
show_result("ragged", function() {
    return ORT\Tensor\Transient::from([[1,2],[3]], ORT\Tensor::INT64);
});

// 3. Mixed types (should fail)
show_result("mixed types", function() {
    return ORT\Tensor\Transient::from([[1,2],[3.5,4]], ORT\Tensor::INT64);
});

// 4. Empty input (should fail)
show_result("empty", function() {
    return ORT\Tensor\Transient::from([], ORT\Tensor::INT64);
});

// 5. Scalar (rank-0) tensor
show_result("scalar", function() {
    return ORT\Tensor\Transient::from([42], ORT\Tensor::INT64);
});

// 6. Boolean tensor
show_result("bool tensor", function() {
    return ORT\Tensor\Transient::from([[true,false],[false,true]], ORT\Tensor::BOOL);
});

// 7. Type not supported (should fail)
show_result("unsupported type", function() {
    return ORT\Tensor\Transient::from([[1,2],[3,4]], 9999);
});
?>
--EXPECTF--
== success 2x2 ==
OK: ORT\Tensor\Transient
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
== ragged ==
EX: ORT\Status\Tensor\InvalidData: ragged array: sub-array at dimension 1 has length 1, expected 2
== mixed types ==
EX: ORT\Status\Tensor\InvalidData: tensor leaf at depth 2: expected integer, got float
== empty ==
EX: ORT\Status\Tensor\InvalidData: empty array encountered at dimension 0 (ragged or empty tensor)
== scalar ==
OK: ORT\Tensor\Transient
array(1) {
  [0]=>
  int(1)
}
== bool tensor ==
OK: ORT\Tensor\Transient
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
== unsupported type ==
EX: ORT\Status\Tensor\InvalidType: unknown data type (9999) provided