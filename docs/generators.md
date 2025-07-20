# Generators

Generators provide a flexible way to produce data on-the-fly for tensor initialization. Instead of requiring static arrays, users can supply objects extending the `Tensor\Generator` abstract to generate values lazily at construction time.

## Motivation

  - Avoid storing large arrays in memory
  - Enable dynamic or randomized tensor initialization
  - Support recursive or complex generation patterns

## Abstract

```php
abstract class Tensor\Generator {
 public function __invoke() : mixed;
}
```

### Usage Examples

#### Simple Generator

```php
class RandomFloatGenerator extends Tensor\Generator {
 public function __invoke() : float {
  return \mt_rand() / \mt_getrandmax();
 }
}
```

#### Recursive Generator

```php
class RecursiveGenerator extends Tensor\Generator {
 public function __invoke() : Tensor\Generator {
  // Return another generator
 }
}
```

`php-ort` will recursively invoke `Tensor\Generator` objects until they resolve
to a usable scalar.

### Tensor Constructors

```Tensor Tensor\Transient(array $shape, Tensor\Generator $generator)```

and

```Tensor Tensor\Persistent(string $name, array $shape, Tensor\Generator $generator)```

### Performance Considerations

While generators offer flexibility and efficiency (in some cases where the alternative is a very large static array which is essentially duplicated on construction), they may introduce some overhead as side effects of lazy evaluation (ie. re-entry to userspace, allocation of call frames etc).

Users should consider performance implications when choosing between using an array initializer or a generator.

At some time, we may provide internal implementations of the most commonly required generators (random numbers and so on), today it's not high priority.

### UX Considerations

The best UX demands that `Tensor\Generator` is an interface because composition > inheritance.

However, the best performance demands that `Tensor\Generator` is an abstract class: When `Tensor\Generator` is abstract, we can use a custom shaped object which contains the fci/fcc for the call, this means we only initialize the fci/fcc once per generator rather than once per call. In addition it provides us a place (the custom shaped object) to store internal pointers required for efficient internal implementation of generators that do not have to allocate call frames for invocation.

In this instance, performance has to be prioritised over purity, so an abstract class is what we have.