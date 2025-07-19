--TEST--
Check tensor getData method with scalar tensors
--EXTENSIONS--
ort
--FILE--
<?php
// Test getData behavior with scalar tensors
try {
    // Create scalar tensors with different data types
    $float_scalar = new ORT\Tensor\Transient([], [3.14], ORT\Tensor::FLOAT);
    $int_scalar = new ORT\Tensor\Transient([], [42], ORT\Tensor::INT64);
    $bool_scalar = new ORT\Tensor\Transient([], [true], ORT\Tensor::BOOL);
    
    // Scalar tensors should return an array with a single element
    $float_data = $float_scalar->getData();
    $int_data = $int_scalar->getData();
    $bool_data = $bool_scalar->getData();
    
    // Using number_format to handle floating point precision issues
    echo "Float scalar: " . gettype($float_data) . ", count: " . count($float_data) . ", value: " . number_format($float_data[0], 2) . "\n";
    echo "Int scalar: " . gettype($int_data) . ", count: " . count($int_data) . ", value: " . $int_data[0] . "\n";
    echo "Bool scalar: " . gettype($bool_data) . ", count: " . count($bool_data) . ", value: " . ($bool_data[0] ? "true" : "false") . "\n";
    
    // Test with parameters to getData
    try {
        // Depth parameter should be invalid for scalar tensors since they have 0 dimensions
        $float_scalar->getData(0, 1);  // Use depth=1 to trigger the exception
        echo "ERROR: Should not accept depth parameter for scalar tensor\n";
    } catch (ORT\Status\Tensor\InvalidShape $e) {
        echo "PASS: Correctly rejected depth parameter for scalar tensor\n";
    }
    
    // Test with offset parameter
    $zero_offset = $float_scalar->getData(0);
    $end_offset = $float_scalar->getData(1);
    
    echo "Zero offset: " . (count($zero_offset) > 0 ? number_format($zero_offset[0], 2) : "empty") . "\n";
    echo "End offset: " . (count($end_offset) > 0 ? "non-empty" : "empty") . "\n";
    
    echo "OK";
} catch(Exception $e) {
    echo "ERROR: " . $e->getMessage();
}
?>
--EXPECT--
Float scalar: array, count: 1, value: 3.14
Int scalar: array, count: 1, value: 42
Bool scalar: array, count: 1, value: true
PASS: Correctly rejected depth parameter for scalar tensor
Zero offset: 3.14
End offset: empty
OK
