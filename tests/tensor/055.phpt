--TEST--
Check Tensor debug info when object is NULL (using reflection)
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test debug info for a tensor where ort->object is NULL
// Try using reflection to create an uninitialized object

echo "=== Testing tensor with NULL object using reflection ===\n";

try {
    // Create tensor object without calling constructor
    $reflectionClass = new ReflectionClass('ONNX\\Tensor\\Persistent');
    $tensor = $reflectionClass->newInstanceWithoutConstructor();
    
    echo "Successfully created tensor object without constructor\n";
    
    // Now try to call var_dump - this should hit the NULL object case
    echo "Calling var_dump on uninitialized tensor:\n";
    var_dump($tensor);
    
} catch (Exception $e) {
    echo "Exception: " . $e->getMessage() . "\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}

?>
--EXPECTF--
=== Testing tensor with NULL object using reflection ===
Successfully created tensor object without constructor
Calling var_dump on uninitialized tensor:
object(ONNX\Tensor\Persistent)#%d (%d) {
}
