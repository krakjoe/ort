--TEST--
Check Tensor with scalar shape (empty shape array)
--EXTENSIONS--
ort
--FILE--
<?php
// Test creating a scalar tensor with empty shape array
try {
    // Create a scalar tensor with a single value
    $scalar = new ONNX\Tensor\Transient(
        [],  // Empty shape array indicates a scalar
        [5], // Single value in data array
        ONNX\Tensor::FLOAT
    );
    
    // Verify properties
    $shape = $scalar->getShape();
    echo "Shape array count: " . count($shape) . "\n";
    
    $data = $scalar->getData();
    echo "Value: " . $data[0] . "\n";
    
    // Verify dimensions
    echo "Is scalar: " . (count($shape) === 0 ? "Yes" : "No") . "\n";
    
    // Also test with different data types
    $int_scalar = new ONNX\Tensor\Transient([], [42], ONNX\Tensor::INT64);
    $bool_scalar = new ONNX\Tensor\Transient([], [true], ONNX\Tensor::BOOL);
    
    echo "Int scalar value: " . $int_scalar->getData()[0] . "\n";
    echo "Bool scalar value: " . ($bool_scalar->getData()[0] ? "true" : "false") . "\n";
    
    // Test persistent scalar tensor
    $persistent = new ONNX\Tensor\Persistent("scalar_test", [], [3.14], ONNX\Tensor::DOUBLE);
    echo "Persistent scalar value: " . $persistent->getData()[0] . "\n";
    
    echo "OK";
} catch(Exception $e) {
    echo "ERROR: " . $e->getMessage();
}
?>
--EXPECT--
Shape array count: 0
Value: 5
Is scalar: Yes
Int scalar value: 42
Bool scalar value: true
Persistent scalar value: 3.14
OK
