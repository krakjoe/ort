--TEST--
ORT Math cores configuration
--EXTENSIONS--
ort
--ENV--
ORT_POOL_CORES=2
--FILE--
<?php
use ORT\Math;

$max = Math\cores();
if ($max != 2) {
    echo "Failed to set cores from environment variable\n";
    exit;
}

$scale = (int) (Math\cores()/2);
$result = Math\scale($scale, function(){
    return [Math\cores(),
            Math\cores(true)];
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