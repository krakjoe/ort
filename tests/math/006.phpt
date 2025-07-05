--TEST--
Check ONNX\Math memory leaks - matmul and sum operations
--EXTENSIONS--
ort
--FILE--
<?php
// Test for memory leaks in matmul and sum operations
try {
    $tensor1 = new ONNX\Tensor\Persistent(
        "matrix_a",
        [2, 2],
        [[2.0, 3.0], [4.0, 5.0]],
        ONNX\Tensor::FLOAT
    );
    
    $tensor2 = new ONNX\Tensor\Persistent(
        "matrix_b",
        [2, 2],
        [[1.0, 2.0], [3.0, 4.0]],
        ONNX\Tensor::FLOAT
    );
    
    // Test matmul
    $result_matmul = ONNX\Math\matmul($tensor1, $tensor2);
    echo "matmul result type: " . get_class($result_matmul) . "\n";
    
    // Test sum
    $result_sum = ONNX\Math\sum($tensor1);
    echo "sum result type: " . get_class($result_sum) . "\n";
    
    echo "Memory leak test passed\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECTF--
matmul result type: ONNX\Tensor\Transient
sum result type: ONNX\Tensor\Transient
Memory leak test passed
