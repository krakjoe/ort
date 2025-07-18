# Generators

Generators provide a flexible way to produce data on-the-fly for tensor initialization. Instead of requiring static arrays, users can supply objects implementing the `Tensor\Generator` interface to generate values lazily at construction time.

## Motivation

  - Avoid storing large arrays in memory
  - Enable dynamic or randomized tensor initialization
  - Support recursive or complex generation patterns

## Interface

```php
interface Tensor\Generator {
 public function __invoke() : mixed;
}
```

### Usage Examples

#### Simple Generator

```php
class RandomFloatGenerator implements Tensor\Generator {
 public function __invoke() : float {
  return \mt_rand() / \mt_getrandmax();
 }
}
```

#### Recursive Generator

```php
class RecursiveGenerator implements Tensor\Generator {
 public function __invoke() : Tensor\Generator {
  // Return another generator
 }
}
```

### Tensor Constructors

```Tensor Tensor\Transient(array $shape, Tensor\Generator $generator)```

and

```Tensor Tensor\Persistent(string $name, array $shape, Tensor\Generator $generator)```


### Performance Considerations

While generators offer flexibility and efficiency (in some cases where the alternative is a very large static array which is essentially duplicated on construction), they may introduce some overhead due to lazy evaluation. Users should consider performance implications when choosing between using an array initializer or a generator.

At some time, we may provide internal implementations of the most commonly required generators (random numbers and so on), today it's not high priority.