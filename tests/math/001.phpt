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
        'ONNX\Math\subtract',
        'ONNX\Math\multiply',
        'ONNX\Math\divide',
        'ONNX\Math\mod',
        'ONNX\Math\pow',
        'ONNX\Math\sqrt',
        'ONNX\Math\cbrt',
        'ONNX\Math\abs',
        'ONNX\Math\neg',
        'ONNX\Math\recip',
        'ONNX\Math\sign',
        'ONNX\Math\ceil',
        'ONNX\Math\floor',
        'ONNX\Math\round',
        'ONNX\Math\trunc',
        'ONNX\Math\sin',
        'ONNX\Math\cos',
        'ONNX\Math\tan',
        'ONNX\Math\asin',
        'ONNX\Math\acos',
        'ONNX\Math\atan',
        'ONNX\Math\sinh',
        'ONNX\Math\cosh',
        'ONNX\Math\tanh',
        'ONNX\Math\exp',
        'ONNX\Math\exp2',
        'ONNX\Math\log',
        'ONNX\Math\log2',
        'ONNX\Math\log10',
        'ONNX\Math\matmul',
        'ONNX\Math\sum',
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
✓ ONNX\Math\subtract
✓ ONNX\Math\multiply
✓ ONNX\Math\divide
✓ ONNX\Math\mod
✓ ONNX\Math\pow
✓ ONNX\Math\sqrt
✓ ONNX\Math\cbrt
✓ ONNX\Math\abs
✓ ONNX\Math\neg
✓ ONNX\Math\recip
✓ ONNX\Math\sign
✓ ONNX\Math\ceil
✓ ONNX\Math\floor
✓ ONNX\Math\round
✓ ONNX\Math\trunc
✓ ONNX\Math\sin
✓ ONNX\Math\cos
✓ ONNX\Math\tan
✓ ONNX\Math\asin
✓ ONNX\Math\acos
✓ ONNX\Math\atan
✓ ONNX\Math\sinh
✓ ONNX\Math\cosh
✓ ONNX\Math\tanh
✓ ONNX\Math\exp
✓ ONNX\Math\exp2
✓ ONNX\Math\log
✓ ONNX\Math\log2
✓ ONNX\Math\log10
✓ ONNX\Math\matmul
✓ ONNX\Math\sum
All core math functions exist!
