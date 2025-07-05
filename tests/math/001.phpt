--TEST--
Check ONNX\Math function availability
--EXTENSIONS--
ort
--FILE--
<?php
try {
    // Test that all ONNX\Math functions exist
    $functions = [
        'ONNX\Math\add',
        'ONNX\Math\multiply', 
        'ONNX\Math\subtract',
        'ONNX\Math\divide',
        'ONNX\Math\sqrt',
        'ONNX\Math\sin',
        'ONNX\Math\cos',
        'ONNX\Math\exp',
        'ONNX\Math\log',
        'ONNX\Math\asin',
        'ONNX\Math\acos',
        'ONNX\Math\atan',
        'ONNX\Math\matmul',
        'ONNX\Math\sum'
    ];
    
    echo "Testing function existence...\n";
    $missing = [];
    foreach ($functions as $func) {
        if (function_exists($func)) {
            echo "✓ $func\n";
        } else {
            $missing[] = $func;
            echo "✗ $func\n";
        }
    }
    
    if (empty($missing)) {
        echo "All core math functions exist!\n";
    } else {
        echo "Missing functions: " . implode(', ', $missing) . "\n";
    }
    
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
Testing function existence...
✓ ONNX\Math\add
✓ ONNX\Math\multiply
✓ ONNX\Math\subtract
✓ ONNX\Math\divide
✓ ONNX\Math\sqrt
✓ ONNX\Math\sin
✓ ONNX\Math\cos
✓ ONNX\Math\exp
✓ ONNX\Math\log
✓ ONNX\Math\asin
✓ ONNX\Math\acos
✓ ONNX\Math\atan
✓ ONNX\Math\matmul
✓ ONNX\Math\sum
All core math functions exist!
