--TEST--
ONNX Math cores configuration
--EXTENSIONS--
ort
--ENV--
ORT_POOL_CORES=2
--FILE--
<?php
if (\ONNX\Math\cores() != 2) {
    echo "Failed to set cores from environment variable\n";
    exit;
}

echo "OK\n";
?>
--EXPECTF--
OK