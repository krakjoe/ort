# API

## Math Functions

#### Arithmetic Operations

```php
function ORT\Math\add(ORT\Tensor $tensor_a, ORT\Tensor|int|float $tensor_b_or_scalar): ORT\Tensor
function ORT\Math\subtract(ORT\Tensor $tensor_a, ORT\Tensor|int|float $tensor_b_or_scalar): ORT\Tensor
function ORT\Math\multiply(ORT\Tensor $tensor_a, ORT\Tensor|int|float $tensor_b_or_scalar): ORT\Tensor
function ORT\Math\divide(ORT\Tensor $tensor_a, ORT\Tensor|int|float $tensor_b_or_scalar): ORT\Tensor
function ORT\Math\mod(ORT\Tensor $tensor_a, ORT\Tensor|int|float $tensor_b_or_scalar): ORT\Tensor
function ORT\Math\neg(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\recip(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\abs(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\sign(ORT\Tensor $tensor): ORT\Tensor
```

#### Exponential and Logarithmic Functions

```php
function ORT\Math\exp(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\exp2(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\log(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\log2(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\log10(ORT\Tensor $tensor): ORT\Tensor
```

#### Power and Root Functions

```php
function ORT\Math\pow(ORT\Tensor $tensor_a, ORT\Tensor|int|float $tensor_b_or_scalar): ORT\Tensor
function ORT\Math\sqrt(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\cbrt(ORT\Tensor $tensor): ORT\Tensor
```

#### Trigonometric Functions

```php
function ORT\Math\sin(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\cos(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\tan(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\asin(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\acos(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\atan(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\arcsin(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\arccos(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\arctan(ORT\Tensor $tensor): ORT\Tensor
```

#### Hyperbolic Functions

```php
function ORT\Math\sinh(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\cosh(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\tanh(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\arcsinh(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\arccosh(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\arctanh(ORT\Tensor $tensor): ORT\Tensor
```

#### Rounding Functions

```php
function ORT\Math\ceil(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\floor(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\round(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\trunc(ORT\Tensor $tensor): ORT\Tensor
```

### Reduction Operations

Operations that reduce tensor dimensions by aggregating values.

#### Full Tensor Reductions

```php
function ORT\Math\reduce\tensor\min(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\reduce\tensor\max(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\reduce\tensor\mean(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\reduce\tensor\sum(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\reduce\tensor\argmax(ORT\Tensor $tensor): ORT\Tensor
function ORT\Math\reduce\tensor\argmin(ORT\Tensor $tensor): ORT\Tensor
```

#### Axis-specific Reductions

```php
function ORT\Math\reduce\axis\min(ORT\Tensor $tensor, int $axis, bool $keepdims = false): ORT\Tensor
function ORT\Math\reduce\axis\max(ORT\Tensor $tensor, int $axis, bool $keepdims = false): ORT\Tensor
function ORT\Math\reduce\axis\mean(ORT\Tensor $tensor, int $axis, bool $keepdims = false): ORT\Tensor
function ORT\Math\reduce\axis\sum(ORT\Tensor $tensor, int $axis, bool $keepdims = false): ORT\Tensor
function ORT\Math\reduce\axis\argmax(ORT\Tensor $tensor, int $axis, bool $keepdims = false): ORT\Tensor
function ORT\Math\reduce\axis\argmin(ORT\Tensor $tensor, int $axis, bool $keepdims = false): ORT\Tensor
```

#### Dot Product

```php
function ORT\Math\reduce\dot(ORT\Tensor $tensor_a, ?ORT\Tensor $tensor_b = null): ORT\Tensor
```

### Transformation Operations

Operations that transform tensor values along an axis.

```php
function ORT\Math\transform\axis\softmax(ORT\Tensor $tensor, int $axis): ORT\Tensor
```

### Matrix Operations

```php
function ORT\Math\matmul(ORT\Tensor $matrix_a, ORT\Tensor $matrix_b): ORT\Tensor
```

### Type Conversion

```php
function ORT\Math\cast(int $type, ORT\Tensor $tensor): ORT\Tensor
```

### Scaling

```php
/**
 * Set scaling parameters
 * @param cores
 * @param code
 * @param threshold
 **/
function ORT\Math\scale(int $cores, callable $code, int $threshold = 0): mixed
/**
 * Get scaling parameter: cores
 * @param max where true, will return the maximum cores available
 **/
function ORT\Math\scale\cores(bool $max = false): int
/**
 * Get scaling parameter: threshold
 * @param default where true, will return the default threshold
 **/
function ORT\Math\scale\threshold(bool $default = false): int
```

### Backend Information

```php
function ORT\Math\backend(): array|false
function ORT\Math\backend\cpu(): string|false
function ORT\Math\backend\gpu(): string|false
```

### Schema Information

```php
class ORT\Math\Schema {
    const int BINARY = 1;
    const int UNARY = 2;
    
    public function __construct(string $symbol);
    public function getSymbol(): ?string;
    public function getKind(): int;
    public function resolve(...$types): int;
}
```

---

## Core Classes

### ORT\Tensor Interface

```php
interface ORT\Tensor {
    const int UNDEFINED;
    const int FLOAT16;
    const int FLOAT32;
    const int FLOAT64;
    const int UINT8;
    const int INT8;
    const int UINT16;
    const int INT16;
    const int UINT32;
    const int INT32;
    const int INT64;
    const int BOOL;
    
    public function isPersistent(): bool;
    public function getName(): ?string;
    public function getType(): int;
    public function getTypeName(): string;
    public function getShape(): array;
    public function getSlice(array $start, array $end, array $axis = []): ORT\Tensor;
    public function getData(int $offset = 0, int $depth = 0): array;
    public function transpose(?array $axis = null): ORT\Tensor;
}
```

### ORT\Tensor\Persistent

```php
class ORT\Tensor\Persistent implements ORT\Tensor {
    public function __construct(
        string $name,
        array $shape = [],
        object|array $data = [],
        int $type = ORT\Tensor::FLOAT32
    );
    
    public static function from(string $name, array $data, int $type): ORT\Tensor;
}
```

### ORT\Tensor\Transient

```php
class ORT\Tensor\Transient implements ORT\Tensor {
    public function __construct(
        array $shape,
        object|array $data,
        int $type = ORT\Tensor::FLOAT32
    );
    
    public static function from(array $data, int $type): ORT\Tensor;
}
```

### ORT\Tensor\Generator\Random

```php
class ORT\Tensor\Generator\Random {
    public function __construct(int $type, int $min = 0, int $max = 1);
    public function __invoke(): mixed;
}
```

### ORT\Model

**Only be available where `ONNXRuntime` is available (ie, `php-ort` was built with `--with-ort-onnx`).**

```php
class ORT\Model {
    public function __construct(
        string $name,
        string $source = '',
        bool $array = false,
        ?ORT\Options $options = null
    );
    
    public static function unload(string $name): bool;
    
    public function getName(): string;
    public function getFile(): string;
    public function getInput(int $index): string;
    public function getInputCount(): int;
    public function getInputIterator(): ORT\Model\Iterator;
    public function getOutput(int $index): string;
    public function getOutputCount(): int;
    public function getOutputIterator(): ORT\Model\Iterator;
    public function getMeta(string $property): mixed;
}
```

### ORT\Runtime

**Only be available where `ONNXRuntime` is available (ie, `php-ort` was built with `--with-ort-onnx`).**

```php
class ORT\Runtime {
    public function __construct(ORT\Model $model);
    public function run(array $input): mixed;
}
```

### ORT\Options

**Only be available where `ONNXRuntime` is available (ie, `php-ort` was built with `--with-ort-onnx`).**

```php
class ORT\Options {
    // Log Levels
    const int LVERBOSE = 0;
    const int LINFO = 1;
    const int LWARNING = 2;
    const int LERROR = 3;
    const int LFATAL = 4;
    
    // Execution Modes
    const int ESEQUENTIAL = 0;
    const int EPARALLEL = 1;
    
    // Graph Optimization Levels
    const int ONONE = 0;
    const int OBASIC = 1;
    const int OEXTENDED = 2;
    const int OALL = 99;
    
    public function __construct();
    public function setExecutionMode(int $mode): void;
    public function setGraphOptimization(int $optimization): void;
    public function setLogId(string $id): void;
    public function setLogSeverity(int $severity): void;
    public function setLogVerbosity(int $verbosity): void;
}
```

---

## Exception Hierarchy

### Status Errors

- `ORT\Status\Error` - Base error class
  - `ORT\Status\Tensor\Error`
    - `ORT\Status\Tensor\NotFound`
    - `ORT\Status\Tensor\InvalidType`
    - `ORT\Status\Tensor\InvalidShape`
    - `ORT\Status\Tensor\InvalidData`
    - `ORT\Status\Tensor\InvalidMemory`
    - `ORT\Status\Tensor\InvalidOffset`
    - `ORT\Status\Tensor\InvalidAccess`
  - `ORT\Status\Math\Error`
    - `ORT\Status\Math\InvalidShape`
    - `ORT\Status\Math\InvalidType`
    - `ORT\Status\Math\InvalidBroadcast`
    - `ORT\Status\Math\InvalidOperation`
    - `ORT\Status\Math\DivisionByZero`
  - `ORT\Status\Model\Error`
    - `ORT\Status\Model\InvalidMemory`
    - `ORT\Status\Model\InvalidOptions`
    - `ORT\Status\Model\InvalidModel`
    - `ORT\Status\Model\InvalidInput`
    - `ORT\Status\Model\InvalidOutput`
    - `ORT\Status\Model\InvalidIndex`
    - `ORT\Status\Model\InvalidFile`
    - `ORT\Status\Model\RuntimeException`
  - `ORT\Status\Schema\Error`
    - `ORT\Status\Schema\InvalidSymbol`
    - `ORT\Status\Schema\InvalidArguments`
  - `ORT\Status\SafetyError`
  - `ORT\Status\ScalingError`