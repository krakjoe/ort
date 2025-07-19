# GPU

**The math library provided by php-ort does not support GPU acceleration directly.**

For the majority of use cases, the performance of SIMD accelerated math provided by `php-ort` will be sufficient.

The lack of direct support is a design choice: GPU integration represents a significant amount of work and commensurate complexity, and the gain is not obviously worth it.

The ONNX Runtime already provides (and maintains) the ability to offload computations to GPUs, and the php-ort extension exposes enough API to access that functionality without burdening this extension with the complexity of its own implementation.

> **Note:** To use GPU acceleration, ONNX Runtime must be installed and configured with GPU support, and your system must have the necessary hardware and drivers.

## Usage

Should you want to perform a particular computation (for example `MatMul`) on a GPU:

  - Create `matmul.onnx` that contains only the `MatMul` node (using Python or another tool).
  - Load the model using `ORT\Model`.
  - Create an instance of `ORT\Runtime`.
    - Create input as usual using a concrete `ORT\Tensor`.
  - Execute the `MatMul` operation on the GPU with `ORT\Runtime::run`.

This methodology can be applied to any operator supported by the ONNX Runtime.

`ORT\Model` objects are persistent (they survive the current request), so during a preloading stage, you may construct all the necessary models your application requires to perform operations on the GPU. In a threaded environment (e.g., FrankenPHP), all threads access the same model (i.e., it is loaded once), but have their own `ORT\Runtime`, allowing for parallel access to the GPU with a single instance of the model loaded into memory.

*Because math is implemented in the namespace `ORT\Math` (not as methods on `Tensor` objects), migrating code to execute on the GPU might be as simple as changing import statements and writing a few functions in your own namespace.*

**Because the `php-ort` extension does not provide any means for creating models (because it's focused on inference), Python or some other tool must be used to create the model; documenting the process here is out of scope.**

## Compliance

The math library provided by php-ort aims to be compliant with the math performed by the ONNX Runtime. This means that as your application scales, you can migrate to executing math on the GPU without needing to change the assumptions made in your code about how math operations behave - you would only need to change how those operations are executed using the methodogy described here.