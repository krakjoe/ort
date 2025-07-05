--TEST--
Large tensor operations to trigger dispatch table and SIMD paths
--EXTENSIONS--
ort
--FILE--
<?php
// Large tensor test to trigger dispatch table and SIMD paths
// This should restore coverage for ort_math_get_dispatch and ort_math_init_dynamic_dispatch_table

echo "=== Large Tensor Coverage Test ===\n";

try {
    $size = 50;  // Above SIMD threshold of 32
    $a = new ONNX\Tensor\Transient([$size], array_fill(0, $size, 1.0), ONNX\Tensor::FLOAT);
    $b = new ONNX\Tensor\Transient([$size], array_fill(0, $size, 2.0), ONNX\Tensor::FLOAT);
    
    // These operations should trigger dispatch table
    $result1 = ONNX\Math\add($a, $b);
    echo "PASS: Large tensor add completed\n";
    
    $result2 = ONNX\Math\multiply($a, $b);
    echo "PASS: Large tensor multiply completed\n";
    
    $result3 = ONNX\Math\subtract($a, $b);
    echo "PASS: Large tensor subtract completed\n";
    
    // Test double precision too
    $a_double = new ONNX\Tensor\Transient([$size], array_fill(0, $size, 1.5), ONNX\Tensor::DOUBLE);
    $b_double = new ONNX\Tensor\Transient([$size], array_fill(0, $size, 2.5), ONNX\Tensor::DOUBLE);
    
    $result4 = ONNX\Math\add($a_double, $b_double);
    echo "PASS: Large tensor double add completed\n";
    
    echo "All large tensor operations completed successfully\n";
    
} catch (Exception $e) {
    echo "FAIL: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
=== Large Tensor Coverage Test ===
PASS: Large tensor add completed
PASS: Large tensor multiply completed
PASS: Large tensor subtract completed
PASS: Large tensor double add completed
All large tensor operations completed successfully
