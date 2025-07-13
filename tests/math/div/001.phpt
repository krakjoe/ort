--TEST--
ONNX\Math\divide: tensor + tensor, all types, shape, numpy/onnx semantics, includes large tensor for vectorization
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = array_merge(range(-16, -1), [0], range(1, 16)); // 33 elements
$unsigned_values = array_merge([0], range(1, 16)); // 17 elements
$types = [
    'FLOAT' => [$real['FLOAT'], $signed_values],
    'DOUBLE' => [$real['DOUBLE'], $signed_values],
    'INT8' => [$signed_types['INT8'], $signed_values],
    'INT16' => [$signed_types['INT16'], $signed_values],
    'INT32' => [$signed_types['INT32'], $signed_values],
    'INT64' => [$signed_types['INT64'], $signed_values],
    'UINT8' => [$unsigned_types['UINT8'], $unsigned_values],
    'UINT16' => [$unsigned_types['UINT16'], $unsigned_values],
    'UINT32' => [$unsigned_types['UINT32'], $unsigned_values],
];
foreach ($types as $name => [$type, $values]) {
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $b = new ONNX\Tensor\Transient([count($values)], $values, $type);
    try {
        $result = ONNX\Math\divide($a, $b);
        echo "PASS: $name divide tensor + tensor\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name divide tensor + tensor: ".$e->getMessage()."\n";
    }
}

// Large tensor case for vectorization
$large_size = 4096;
foreach ($types as $name => [$type, $values]) {
    $a = new ONNX\Tensor\Transient([$large_size], array_fill(0, $large_size, 2), $type);
    $b = new ONNX\Tensor\Transient([$large_size], array_fill(0, $large_size, 2), $type);
    try {
        $result = ONNX\Math\divide($a, $b);
        $data = $result->getData();
        echo "PASS: $name divide large tensor + large tensor (vectorized)\n";
        printf("RESULT: [%d x %d] first=%.1f last=%.1f\n", $large_size, $large_size, $data[0], $data[$large_size-1]);
        printf("TYPE: %s\n", $result->getTypeName());
        printf("SHAPE: [%d]\n", $large_size);
    } catch (Throwable $e) {
        echo "FAIL: $name divide large tensor + large tensor: ".$e->getMessage()."\n";
    }
}
?>
--EXPECTF--
PASS: FLOAT divide tensor + tensor
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT16 divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT32 divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT64 divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: UINT8 divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [17]
PASS: UINT16 divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [17]
PASS: UINT32 divide tensor + tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [17]
PASS: FLOAT divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: FLOAT
SHAPE: [4096]
PASS: DOUBLE divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
PASS: INT8 divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
PASS: INT16 divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
PASS: INT32 divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
PASS: INT64 divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
PASS: UINT8 divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
PASS: UINT16 divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
PASS: UINT32 divide large tensor + large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
