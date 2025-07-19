--TEST--
Check Status\Tensor\InvalidData - invalid data for scalar tensor
--EXTENSIONS--
ort
--FILE--
<?php
try {
    // Empty shape array expects a scalar value (single element), not an array of values
    new ORT\Tensor\Persistent(
        "invalid_tensor",
        [],
        [1, 2, 3],  // This is wrong - for scalar tensors we need a single value like [5]
        ORT\Tensor::INT64
    );
    echo "FAIL: Should reject multi-element data for scalar tensor\n";
} catch(ORT\Status\Tensor\InvalidData $e) {
    echo "OK: Correctly rejected invalid data for scalar tensor\n";
} catch(Exception $e) {
    echo "ERROR: Caught unexpected exception: " . get_class($e) . "\n";
    echo $e->getMessage() . "\n";
}

// This should work - proper scalar tensor
try {
    $scalar = new ORT\Tensor\Persistent(
        "valid_scalar",
        [],  // Empty shape array = scalar tensor
        [42], // Single value for scalar tensor
        ORT\Tensor::INT64
    );
    echo "OK: Created valid scalar tensor\n";
} catch(Exception $e) {
    echo "FAIL: Valid scalar tensor creation failed: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
OK: Correctly rejected invalid data for scalar tensor
OK: Created valid scalar tensor