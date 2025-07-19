--TEST--
Check scalar tensor mathematical operations
--EXTENSIONS--
ort
--FILE--
<?php
// Test scalar tensor operations with math functions
try {
    // Create scalar tensors
    $scalar1 = new ORT\Tensor\Transient([], [4], ORT\Tensor::FLOAT);
    $scalar2 = new ORT\Tensor\Transient([], [2], ORT\Tensor::FLOAT);
    
    // Test basic operations
    $sum = ORT\Math\add($scalar1, $scalar2);
    $diff = ORT\Math\subtract($scalar1, $scalar2);
    $prod = ORT\Math\multiply($scalar1, $scalar2);
    $quot = ORT\Math\divide($scalar1, $scalar2);
    
    echo "Sum: " . $sum->getData()[0] . "\n";
    echo "Difference: " . $diff->getData()[0] . "\n";
    echo "Product: " . $prod->getData()[0] . "\n";
    echo "Quotient: " . $quot->getData()[0] . "\n";
    
    // Test unary operations
    $sqrt = ORT\Math\sqrt($scalar1);
    $neg = ORT\Math\neg($scalar1);
    
    echo "Square root: " . $sqrt->getData()[0] . "\n";
    echo "Negation: " . $neg->getData()[0] . "\n";
    
    // Test scalar-tensor broadcasting
    $vector = new ORT\Tensor\Transient([3], [1, 2, 3], ORT\Tensor::FLOAT);
    $scaled = ORT\Math\multiply($vector, $scalar2);
    
    $scaled_data = $scaled->getData();
    echo "Scaled vector: [" . implode(", ", $scaled_data) . "]\n";
    
    echo "OK";
} catch(Exception $e) {
    echo "ERROR: " . $e->getMessage();
}
?>
--EXPECT--
Sum: 6
Difference: 2
Product: 8
Quotient: 2
Square root: 2
Negation: -4
Scaled vector: [2, 4, 6]
OK
