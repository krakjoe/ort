--TEST--
ORT\Math\Schema: test constructor (missing schemas)
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Math;

function get_defined_function_name($symbol) {
    $parts = \explode("\\", $symbol);
    $name = \end($parts);
    switch ($name) {
        case "multiply":
            return "mul";
        case "divide":
            return "div";
        case "subtract":
            return "sub"; 
    }
    return $name;
}

foreach (get_defined_functions()["internal"] as $symbol) {
    if (substr($symbol, 0, 3) != "ort") {
        continue;
    }

    if ($symbol == "ort\\math\\cast") {
        continue;
    }

    if (substr($symbol, 0,
            strlen("ort\\math\\backend")) ==
                "ort\\math\\backend") {
        continue;
    }

    if (substr($symbol, 0,
            strlen("ort\\math\\scale")) ==
                "ort\\math\\scale") {
        continue;
    }

    new Math\Schema(get_defined_function_name($symbol));
}

echo "OK";
?>
--EXPECT--
OK