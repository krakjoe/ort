--TEST--
ORT\Math\Schema: invalid symbol
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Math;
use ORT\Status;

try {
    new Math\Schema("noSuchFunctionAndThereNeverWillBe");
} catch (Status\Schema\InvalidSymbol $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECT--
string(60) "the symbol noSuchFunctionAndThereNeverWillBe cannot be found"