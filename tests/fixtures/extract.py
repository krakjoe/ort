#!/usr/bin/env python3
import numpy as np
import argparse
import sys

TYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.bool,
]

C_ENUM = {
    'float16': 'ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT',
    'float32': 'ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT',
    'float64': 'ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE',
    'int8':    'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8',
    'int16':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16',
    'int32':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32',
    'int64':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64',
    'uint8':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8',
    'uint16':  'ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16',
    'uint32':  'ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32',
    'bool':    'ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL',
}

def type_name(dtype):
    return str(np.dtype(dtype))

def get_func(name):
    try:
        return getattr(np, name)
    except AttributeError:
        print(f"Function '{name}' not found in numpy. Available functions: {', '.join(dir(np))}", file=sys.stderr)
        sys.exit(1)

def emit_schema_indices(name, size, indices):
        print(f"static const ONNXTensorElementDataType ort_math_promotion_schema_indices_{name}[{size}] = {{")
        for t in indices:
            print(f"    {C_ENUM[t]},")
        print("};\n")

def emit_schema_struct(name, size):
    print(f"static const ort_math_type_promotion_schema_t ort_math_promotion_schema_{name} = {{")
    print(f"    .table   = ort_math_promotion_schema_table_{name},")
    print(f"    .indices = ort_math_promotion_schema_indices_{name},")
    print(f"    .size    = {size}")
    print("};")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function', required=True, help='Function name (e.g. add, multiply)')
    parser.add_argument('-b', '--binary',   action='store_true', help='Generate binary promotion schema')
    parser.add_argument('-u', '--unary',    action='store_true', help='Generate unary promotion schema')
    parser.add_argument('-n', '--name',     default='onnx', help='Schema name suffix')
    args = parser.parse_args()

    if args.unary and args.binary:
        print("Error: Cannot generate both unary and binary promotion schemas at the same time.", file=sys.stderr)
        sys.exit(1)

    func       = get_func(args.function)
    size       = len(TYPES)
    type_names = [type_name(t) for t in TYPES]

    if args.binary:
        print(f"static const ONNXTensorElementDataType ort_math_promotion_schema_table_{args.name}[{size}*{size}] = {{")
        for i, t1 in enumerate(TYPES):
            row = []
            for j, t2 in enumerate(TYPES):
                try:
                    a = np.array([1], dtype=t1)
                    b = np.array([1], dtype=t2)
                    result = func(a, b)
                    res_type = type_name(result.dtype)
                except Exception:
                    res_type = 'UNDEFINED'
                row.append(f"    {C_ENUM.get(res_type, 'ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED')},\t// {type_names[i]} {args.function} {type_names[j]} -> {res_type}\n")
            print("" + "".join(row))
        print("};\n")
        emit_schema_indices(args.name, size, type_names)
        emit_schema_struct(args.name, size)

    if args.unary:
        print(f"static const ONNXTensorElementDataType ort_math_promotion_schema_table_{args.name}[{size}] = {{")
        for i, t in enumerate(TYPES):
            try:
                a = np.array([1], dtype=t)
                result = func(a)
                res_type = type_name(result.dtype)
            except Exception:
                res_type = 'UNDEFINED'
            print(f"    {C_ENUM.get(res_type, 'ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED')},\t// {type_names[i]} -> {res_type}")
        print("};\n")
        emit_schema_indices(args.name, size, type_names)
        emit_schema_struct(args.name, size)


if __name__ == '__main__':
    main()