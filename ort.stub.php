<?php

/**
 * @generate-class-entries
 * @generate-legacy-arginfo 80000
 */

namespace ORT {
    /**
     * @var int
     * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
     */
    const TENSOR_UNDEFINED = UNKNOWN;

    interface Tensor
    {
        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
         */
        public const UNDEFINED = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
         */
        public const FLOAT16 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32
         */
        public const FLOAT32 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64
         */
        public const FLOAT64 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
         */
        public const UINT8 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
         */
        public const INT8 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
         */
        public const UINT16 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
         */
        public const INT16 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
         */
        public const UINT32 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
         */
        public const INT32 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
         */
        public const INT64 = UNKNOWN;

        /**
         * @var int
         * @cvalue ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
         */
        public const BOOL = UNKNOWN;

        public function isPersistent(): bool;

        public function getName(): ?string;

        public function getType(): int;

        public function getTypeName(): string;

        public function getShape(): array;

        public function getSlice(array $start, array $end, array $axis = []): Tensor;

        public function getData(int $offset = 0, int $depth = 0): array;

        public function transpose(?array $axis = null): Tensor;
    }

    /**
     * @not-serializable
     */
    class Model
    {
        public function __construct(
            string $name,
            string $source = '',
            bool $array = false,
            ?Options $options = null
        ) {}

        public static function unload(string $name): bool {}

        public function getName(): string {}

        public function getFile(): string {}

        public function getInput(int $index): string {}

        public function getInputCount(): int {}

        /** @return \ORT\Model\Iterator */
        public function getInputIterator(): Model\Iterator {}

        public function getOutput(int $index): string {}

        public function getOutputCount(): int {}

        /** @return \ORT\Model\Iterator */
        public function getOutputIterator(): Model\Iterator {}

        public function getMeta(string $property): mixed {}
    }

    /**
     * @not-serializable
     */
    class Runtime
    {
        public function __construct(Model $model) {}

        public function run(array $input): mixed {}
    }

    /**
     * @not-serializable
     */
    class Options
    {
        /**
         * @var int
         * @cvalue ORT_LOGGING_LEVEL_VERBOSE
         */
        public const LVERBOSE = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_LOGGING_LEVEL_INFO
         */
        public const LINFO = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_LOGGING_LEVEL_WARNING
         */
        public const LWARNING = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_LOGGING_LEVEL_ERROR
         */
        public const LERROR = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_LOGGING_LEVEL_FATAL
         */
        public const LFATAL = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_SEQUENTIAL
         */
        public const ESEQUENTIAL = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_PARALLEL
         */
        public const EPARALLEL = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_DISABLE_ALL
         */
        public const ONONE = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_ENABLE_BASIC
         */
        public const OBASIC = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_ENABLE_EXTENDED
         */
        public const OEXTENDED = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_ENABLE_ALL
         */
        public const OALL = UNKNOWN;

        public function __construct() {}

        public function setExecutionMode(int $mode): void {}

        public function setGraphOptimization(int $optimization): void {}

        public function setLogId(string $id): void {}

        public function setLogSeverity(int $severity): void {}

        public function setLogVerbosity(int $verbosity): void {}
    }
}

namespace ORT\Tensor {
    /**
     * @not-serializable
     */
    class Persistent implements \ORT\Tensor
    {
        public function __construct(
            string $name,
            array $shape = [],
            array|object $data = [],
            int $type = \ORT\Tensor::FLOAT32
        ) {}

        public static function from(string $name, array $data, int $type): \ORT\Tensor {}

        public function isPersistent(): bool {}

        public function getName(): ?string {}

        public function getType(): int {}

        public function getTypeName(): string {}

        public function getShape(): array {}

        public function getSlice(array $start, array $end, array $axis = []): \ORT\Tensor {}

        public function getData(int $offset = 0, int $depth = 0): array {}

        public function transpose(?array $axis = null): \ORT\Tensor {}
    }

    /**
     * @not-serializable
     */
    class Transient implements \ORT\Tensor
    {
        public function __construct(
            array $shape,
            array|object $data,
            int $type = \ORT\Tensor::FLOAT32
        ) {}

        public static function from(array $data, int $type): \ORT\Tensor {}

        public function isPersistent(): bool {}

        public function getName(): ?string {}

        public function getType(): int {}

        public function getTypeName(): string {}

        public function getShape(): array {}

        public function getSlice(array $start, array $end, array $axis = []): \ORT\Tensor {}

        public function getData(int $offset = 0, int $depth = 0): array {}

        public function transpose(?array $axis = null): \ORT\Tensor {}
    }

    /**
     * Base class for tensor data generators.
     * Implement __invoke() to provide custom generation logic.
     */
    abstract class Generator
    {
        abstract public function __invoke(): mixed;
    }
}

namespace ORT\Tensor\Generator {
    /**
     * Random number generator for tensors.
     * Generates random values within specified bounds based on tensor type.
     */
    class Random extends \ORT\Tensor\Generator
    {
        public function __construct(int $type, int|float $min = 0, int|float $max = 1) {}

        public function __invoke(): mixed {}
    }
}

namespace ORT\Model {
    /**
     * Iterator for model input/output names.
     */
    class Iterator implements \Iterator
    {
        public function current(): string {}

        public function key(): int {}

        public function next(): void {}

        public function rewind(): void {}

        public function valid(): bool {}
    }
}

namespace ORT\Math {
    /**
     * Schema for type promotion in mathematical operations.
     */
    final class Schema
    {
        /**
         * @var int
         * @cvalue ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY
         */
        public const BINARY = UNKNOWN;

        /**
         * @var int
         * @cvalue ORT_MATH_PROMOTION_SCHEMA_KIND_UNARY
         */
        public const UNARY = UNKNOWN;

        public function __construct(string $symbol) {}

        public function getSymbol(): ?string {}

        public function getKind(): int {}

        public function resolve(int ...$types): int {}
    }

    // -------------------------------------------------------------------------
    // Arithmetic Operations
    // -------------------------------------------------------------------------

    function add(\ORT\Tensor $tensor_a, \ORT\Tensor|int|float $tensor_b_or_scalar): \ORT\Tensor {}

    function subtract(\ORT\Tensor $tensor_a, \ORT\Tensor|int|float $tensor_b_or_scalar): \ORT\Tensor {}

    function multiply(\ORT\Tensor $tensor_a, \ORT\Tensor|int|float $tensor_b_or_scalar): \ORT\Tensor {}

    function divide(\ORT\Tensor $tensor_a, \ORT\Tensor|int|float $tensor_b_or_scalar): \ORT\Tensor {}

    function mod(\ORT\Tensor $tensor_a, \ORT\Tensor|int|float $tensor_b_or_scalar): \ORT\Tensor {}

    function pow(\ORT\Tensor $tensor_a, \ORT\Tensor|int|float $tensor_b_or_scalar): \ORT\Tensor {}

    function neg(\ORT\Tensor $tensor): \ORT\Tensor {}

    function recip(\ORT\Tensor $tensor): \ORT\Tensor {}

    function abs(\ORT\Tensor $tensor): \ORT\Tensor {}

    function sign(\ORT\Tensor $tensor): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Exponential and Logarithmic Functions
    // -------------------------------------------------------------------------

    function exp(\ORT\Tensor $tensor): \ORT\Tensor {}

    function exp2(\ORT\Tensor $tensor): \ORT\Tensor {}

    function log(\ORT\Tensor $tensor): \ORT\Tensor {}

    function log2(\ORT\Tensor $tensor): \ORT\Tensor {}

    function log10(\ORT\Tensor $tensor): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Power and Root Functions
    // -------------------------------------------------------------------------

    function sqrt(\ORT\Tensor $tensor): \ORT\Tensor {}

    function cbrt(\ORT\Tensor $tensor): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Trigonometric Functions
    // -------------------------------------------------------------------------

    function sin(\ORT\Tensor $tensor): \ORT\Tensor {}

    function cos(\ORT\Tensor $tensor): \ORT\Tensor {}

    function tan(\ORT\Tensor $tensor): \ORT\Tensor {}

    function asin(\ORT\Tensor $tensor): \ORT\Tensor {}

    function acos(\ORT\Tensor $tensor): \ORT\Tensor {}

    function atan(\ORT\Tensor $tensor): \ORT\Tensor {}

    function arcsin(\ORT\Tensor $tensor): \ORT\Tensor {}

    function arccos(\ORT\Tensor $tensor): \ORT\Tensor {}

    function arctan(\ORT\Tensor $tensor): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Hyperbolic Functions
    // -------------------------------------------------------------------------

    function sinh(\ORT\Tensor $tensor): \ORT\Tensor {}

    function cosh(\ORT\Tensor $tensor): \ORT\Tensor {}

    function tanh(\ORT\Tensor $tensor): \ORT\Tensor {}

    function arcsinh(\ORT\Tensor $tensor): \ORT\Tensor {}

    function arccosh(\ORT\Tensor $tensor): \ORT\Tensor {}

    function arctanh(\ORT\Tensor $tensor): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Rounding Functions
    // -------------------------------------------------------------------------

    function ceil(\ORT\Tensor $tensor): \ORT\Tensor {}

    function floor(\ORT\Tensor $tensor): \ORT\Tensor {}

    function round(\ORT\Tensor $tensor): \ORT\Tensor {}

    function trunc(\ORT\Tensor $tensor): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Matrix Operations
    // -------------------------------------------------------------------------

    function matmul(\ORT\Tensor $matrix_a, \ORT\Tensor $matrix_b): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Type Conversion
    // -------------------------------------------------------------------------

    function cast(int $type, \ORT\Tensor $tensor): \ORT\Tensor {}

    // -------------------------------------------------------------------------
    // Scaling and Backend Information
    // -------------------------------------------------------------------------

    function scale(int $cores, callable $code, int $threshold = 0): mixed {}

    function backend(): array|false {}
}

namespace ORT\Math\reduce\tensor {
    function min(\ORT\Tensor $tensor): \ORT\Tensor {}

    function max(\ORT\Tensor $tensor): \ORT\Tensor {}

    function mean(\ORT\Tensor $tensor): \ORT\Tensor {}

    function sum(\ORT\Tensor $tensor): \ORT\Tensor {}

    function argmax(\ORT\Tensor $tensor): \ORT\Tensor {}

    function argmin(\ORT\Tensor $tensor): \ORT\Tensor {}
}

namespace ORT\Math\reduce\axis {
    function min(\ORT\Tensor $tensor, int $axis, bool $keepdims = false): \ORT\Tensor {}

    function max(\ORT\Tensor $tensor, int $axis, bool $keepdims = false): \ORT\Tensor {}

    function mean(\ORT\Tensor $tensor, int $axis, bool $keepdims = false): \ORT\Tensor {}

    function sum(\ORT\Tensor $tensor, int $axis, bool $keepdims = false): \ORT\Tensor {}

    function argmax(\ORT\Tensor $tensor, int $axis, bool $keepdims = false): \ORT\Tensor {}

    function argmin(\ORT\Tensor $tensor, int $axis, bool $keepdims = false): \ORT\Tensor {}
}

namespace ORT\Math\reduce {
    function dot(\ORT\Tensor $tensor_a, \ORT\Tensor $tensor_b): \ORT\Tensor {}
}

namespace ORT\Math\transform\axis {
    function softmax(\ORT\Tensor $tensor, int $axis): \ORT\Tensor {}
}

namespace ORT\Math\scale {
    function cores(bool $max = false): int {}

    function threshold(bool $default = false): int {}
}

namespace ORT\Math\backend {
    function cpu(): string|false {}

    function gpu(): string|false {}
}

// =============================================================================
// Exception Hierarchy
// =============================================================================

namespace ORT\Status {
    class Error extends \Error {}

    class SafetyError extends Error {}

    class ScalingError extends Error {}
}

namespace ORT\Status\Tensor {
    class Error extends \ORT\Status\Error {}

    class NotFound extends \ORT\Status\Tensor\Error {}

    class InvalidType extends \ORT\Status\Tensor\Error {}

    class InvalidShape extends \ORT\Status\Tensor\Error {}

    class InvalidData extends \ORT\Status\Tensor\Error {}

    class InvalidMemory extends \ORT\Status\Tensor\Error {}

    class InvalidOffset extends \ORT\Status\Tensor\Error {}

    class InvalidAccess extends \ORT\Status\Tensor\Error {}
}

namespace ORT\Status\Math {
    class Error extends \ORT\Status\Error {}

    class InvalidShape extends \ORT\Status\Math\Error {}

    class InvalidType extends \ORT\Status\Math\Error {}

    class InvalidBroadcast extends \ORT\Status\Math\Error {}

    class InvalidOperation extends \ORT\Status\Math\Error {}

    class DivisionByZero extends \ORT\Status\Math\Error {}
}

namespace ORT\Status\Model {
    class Error extends \ORT\Status\Error {}

    class InvalidMemory extends \ORT\Status\Model\Error {}

    class InvalidOptions extends \ORT\Status\Model\Error {}

    class InvalidModel extends \ORT\Status\Model\Error {}

    class InvalidInput extends \ORT\Status\Model\Error {}

    class InvalidOutput extends \ORT\Status\Model\Error {}

    class InvalidIndex extends \ORT\Status\Model\Error {}

    class InvalidFile extends \ORT\Status\Model\Error {}

    class RuntimeException extends \ORT\Status\Model\Error {}
}

namespace ORT\Status\Schema {
    class Error extends \ORT\Status\Error {}

    class InvalidSymbol extends \ORT\Status\Schema\Error {}

    class InvalidArguments extends \ORT\Status\Schema\Error {}
}
