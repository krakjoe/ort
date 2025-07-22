--TEST--
ORT\Math\scale\cores
--EXTENSIONS--
ort
--ENV--
ORT_SCALE_CORES=2
--FILE--
<?php
use ORT\Math;

$max = Math\scale\cores();
if ($max != 2) {
    echo "Failed to set cores from environment variable\n";
    exit;
}

$scale = (int) (Math\scale\cores()/2);
$result = Math\scale($scale, function(){
    return [Math\scale\cores(),
            Math\scale\cores(true)];
});

if ($result[0] != $scale) {
    echo "Failed to scale cores\n";
    var_dump($result);
    exit;
}

if ($result[1] != $max) {
    echo "Max cores incorrectly reported while scaling\n";
    var_dump($result);
    exit;
}

echo "OK\n";
?>
--EXPECTF--
OK