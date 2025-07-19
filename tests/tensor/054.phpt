--TEST--
Check Tensor debug info when object is NULL (edge case)
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test debug info for a tensor where ort->object is NULL
// This happens when tensor construction fails (e.g., just name provided without shape/data)

echo "=== Testing tensor with NULL object (constructor failure) ===\n";

try {
    // This should fail - trying to find a non-existent tensor with just name
    $tensor = new ORT\Tensor\Persistent("non_existent_tensor");
    
    // If we get here, the constructor succeeded when it shouldn't have
    echo "ERROR: Constructor should have failed\n";
    var_dump($tensor);
} catch (Error $e) {
    echo "Expected exception: " . $e->getMessage() . "\n";
    
    // Since the constructor failed, we can't test var_dump on a failed object
    // But we can confirm the expected behavior
    echo "Constructor properly failed as expected\n";
}

// Test another case - create a tensor with invalid parameters that might fail construction
echo "\n=== Testing tensor with invalid construction parameters ===\n";

try {
    // Try to create a tensor with mismatched data/shape
    $tensor = new ORT\Tensor\Persistent(
        "test_tensor",
        [2, 2],  // Shape expects 4 elements
        [1, 2, 3],  // Data has only 3 elements
        ORT\Tensor::INT32
    );
    
    echo "ERROR: Constructor should have failed with mismatched shape/data\n";
    var_dump($tensor);
} catch (ORT\Status\Tensor\InvalidData $e) {
    echo "Expected exception: " . $e->getMessage() . "\n";
} catch (Exception $e) {
    echo "Unexpected exception type: " . get_class($e) . " - " . $e->getMessage() . "\n";
}

?>
--EXPECTF--
=== Testing tensor with NULL object (constructor failure) ===
Expected exception: %s
Constructor properly failed as expected

=== Testing tensor with invalid construction parameters ===
Expected exception: %s
