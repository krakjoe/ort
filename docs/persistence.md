# Persistence

Tensor objects can become quite large, because PHP has a shared nothing architecture this could become a problem.

# ONNX\Tensor Opt-In Persistence

`ONNX\Tensor` is an interface implemented by the concrete classes:

  - `ONNX\Tensor\Persistent`
  - `ONNX\Tensor\Transient`

A persistent tensor is named upon creation, and will survive the request lifecycle, subsequent requests need only know the name of the tensor to instantiate it.

A transient tensor has no name, may be returned by math operations and is inaccessible from other contexts.

Many *threads* may work on the same persistent tensor concurrently, the API provides read only views via `Tensor::getSlice`.

# ONNX\Model Persistence

Models are persistent (they survive the request cycle) and shared among contexts (so far as the current processing model allows[1]).

Again they are named upon creation, and their name and source file must be known to retrieve them.

Many *threads* may work with the same model concurrently.

## Warm Up

In a typical deployment, one should warm up the worker (in the case of FPM) or process (in the case of FrankenPHP) by creating any large models, and persistent tensors that are required in subsequent requests.

### Processing Models

[1] In non-threaded environments such as Apache MPM prefork, one would have to deploy preloading to create persistent tensors and models in order for those to be shared among all processes. In a threaded environment like FrankenPHP, sharing among worker threads is allowed (and safe).