--TEST--
ORT\Math\scale\threshold
--EXTENSIONS--
ort
--ENV--
ORT_SCALE_CORES=2
ORT_SCALE_THRESHOLD=42
--FILE--
<?php
use ORT\Math;

$threshold = Math\scale\threshold();
if ($threshold != 42) {
    echo "Failed to set threshold from environment variable\n";
    exit;
}

$check = (int) (Math\scale\threshold()/2);

$result = Math\scale(Math\scale\cores(), function(){
    return [Math\scale\threshold(),
            Math\scale\threshold(true)];
}, $check);

if ($result[0] != $check) {
    echo "Failed to scale threshold\n";
    var_dump($result);
    exit;
}

if ($result[1] != Math\scale\threshold(true)) {
    echo "Default threshold incorrectly reported while scaling\n";
    var_dump($result);
    exit;
}

echo "OK\n";
?>
--EXPECTF--
OK