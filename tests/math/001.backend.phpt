--TEST--
ORT Math backend configuration
--EXTENSIONS--
ort
--FILE--
<?php
$backend = \ORT\Math\backend();

if ($backend === false) {
    printf("Backend: none\n");
} else if ($backend) {
    printf("Backend: %s\n", $backend);
} else {
    printf("backend() returned something unexpected");
}
?>
--EXPECTF--
Backend: %s