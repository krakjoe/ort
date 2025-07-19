--TEST--
ORT\Tensor\Generator
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

$generated = new Tensor\Transient([], new class
    implements Tensor\Generator {
    public function __invoke() : int {
        return 42;
    }
});

printf("GENERATED SCALAR: %d\n", $generated->getData()[0]);

$generated = new Tensor\Transient([], new class
    implements Tensor\Generator {
    public function __invoke() : Tensor\Generator {
        return new class 
            implements Tensor\Generator {
            public function __invoke() : int {
                return 42;
            }
        };
    }
});

printf("TURTLES: %d\n", $generated->getData()[0]);

$generator = new class
    implements Tensor\Generator {

    public $counter = 0;

    public function __invoke() : float {
        return $this->counter++;
    }
};

$generated = new Tensor\Transient([16, 16], $generator, Tensor::FLOAT);

printf("GENERATED MATRIX: %d\n", $generator->counter);
?>
--EXPECT--
GENERATED SCALAR: 42
TURTLES: 42
GENERATED MATRIX: 256