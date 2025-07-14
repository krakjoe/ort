# Type Promotions

Type promotions in `php-ort` are context dependent: This means that coercion will occur depending on the call rather than the data. The schema for promotion is extracted from numpy's runtime and exported to C in `src/maths/schema`. This allows `php-ort` to rigidly conform to `numpy` behaviour in a uniform way across all operators and types.

# Extracting Promotion Schema

  - Script: `tests/fixtures/extract.py`
  - Arguments:
    - `-f`/`--function`: The name of the function you want to extract the schema for.
    - `-c`/`--code`:     Code to execute instead of `function` [1]
    - `-u`/`--unary`:    The function must be invoked as a unary function.
    - `-b`/`--binary`:   The function must be invoked as a binary function.
    - `-n`/`--name`:     The suffix of the name of the struct (should usually be function name).
  - Exclusivity:
    - `f`/`c`: Only one of `function` or `code` may be used.
    - `u`/`b`: Only one of `unary` or `binary` may be used.

## Example

`python3 tests/fixtures/extract.py -f sqrt -n sqrt -u`

Will extract the promotion schema for `np.sqrt` (which is a unary function). The script will output the C code for the schema, which should be copy/pasted into the appropriate schema header in `src/maths/schema` and included from the applicable frontend unit (in this case `sqrt.c`).

## Introspection

Introspection of promotion schemas can be performed using the `ONNX\Math\Schema` API:

  - `ONNX\Math\Schema::__construct(string symbol)`
  - `ONNX\Math\Schema::resolve(int ... types)`

Example:

```php
$schema = new ONNX\Math\Schema('add');

var_dump($schema->resolve(
    ONNX\Tensor::FLOAT,
    ONNX\Tensor::DOUBLE) ==
        ONNX\Tensor::DOUBLE);
```

Will yield:

```text
bool(true)
```

Because adding `Tensor::FLOAT` and `Tensor::DOUBLE` results in `Tensor::DOUBLE`.

Errors: `resolve` will return `-1` upon encountering error conditions (adjust your API usage).

## Casting

Casting between types is handled implicitly according to the promotion schema. For example, if we have two tensors of type `float` and `double`, then they will automatically be coerced to `double` before being passed to the kernel. This happens transparently and just in time, however it's sometimes preferable to coerce explicitly (which we call casting), because this makes code easier to reason about, and possibly more efficient.

`ONNX\Tensor ONNX\Math\cast(int $type, ONNX\Tensor $tensor)` shall cast `$tensor` to `$type` and return the resulting `Tensor`.

**Whether explicit or implicit, `php-ort` parallelizes large casting operations automatically.**

### Notes

[1] `code` should be used in cases where function and operator behavior are not consistent in `numpy`. This is only true in some edge cases (like `recip`). In these cases, `php-ort` should prefer the operator behavior over the function behavior and so provide a lambda implementing the expression.

#### Technical Notes

  - Promotion schemas drive both coercion behavior and kernel selection.
  - The inclusion of `UNDEFINED` in the output is not an error, but indicates that the operation is not mathmatically defined in `numpy` and so should not be supported by `php-ort` (ie, don't hand edit schemas with `UNDEFINED`).
  - Remember to edit `src/maths.c` for new schemas; introspection requires manual edits to this file.