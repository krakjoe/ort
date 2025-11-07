#!/usr/bin/env python3
import os
import numpy as np
import argparse
import sys
import shlex

EX_ORT_TYPES = {
    'float16': 'ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16',
    'float32': 'ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32',
    'float64': 'ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64',
    'int8':    'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8',
    'int16':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16',
    'int32':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32',
    'int64':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64',
    'uint8':   'ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8',
    'uint16':  'ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16',
    'uint32':  'ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32',
    'bool':    'ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL',
}

EX_NUMPY_TYPES = [
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

def ex_get_type(dtype):
    return str(np.dtype(dtype))

def ex_get_callable(args):
    if args.code:
        try:
            callable = eval(args.code, {"np": np})
        except Exception as e:
            print(f"Error: evaluating code failed: {e}",
                file=sys.stderr)
            sys.exit(1)
    else:
        try:
            callable = getattr(np, args.function)
        except AttributeError:
            print(f"Error: function '{args.function}' not found in numpy. "
                  f"Available functions: {', '.join(dir(np))}",
                  file=sys.stderr)
            sys.exit(1)

    return callable

def ex_emit_name(name, template):
    return template.replace('{NAME}', name.upper())

def ex_emit_extract(name, args, template):
    schema = f"%python% %extract.py% {' '.join(
        shlex.quote(arg) for arg in args[1:])}"
    return template.replace('{EXTRACT}', schema)

def ex_emit_indices(name, size, indices, template):
    schema = f"static const ONNXTensorElementDataType ort_math_promotion_schema_indices_{name}[{size}] = {{\n"
    for t in indices:
        schema += (f"    {EX_ORT_TYPES[t]},\n")
    schema += ("};")
    return template.replace('{INDICES}', schema)

def ex_emit_struct(name, size, kind, operands, template):
    schema = f"static const ort_math_promotion_schema_t ort_math_promotion_schema_{name} = {{\n"
    schema += (f"    .kind     = {kind},\n")
    schema += (f"    .operands = {operands},\n")
    schema += (f"    .table    = ort_math_promotion_schema_table_{name},\n")
    schema += (f"    .indices  = ort_math_promotion_schema_indices_{name},\n")
    schema += (f"    .size     = {size}\n")
    schema += ("};")
    return template.replace('{STRUCT}', schema)

def ex_emit_binary(name, size, types, callable, template):
    schema = f"static const ONNXTensorElementDataType ort_math_promotion_schema_table_{name}[{size}*{size}] = {{\n"
    for i, t1 in enumerate(EX_NUMPY_TYPES):
        row = []
        for j, t2 in enumerate(EX_NUMPY_TYPES):
            try:
                a = np.array([1], dtype=t1)
                b = np.array([1], dtype=t2)
                result = callable(a, b)
                res_type = ex_get_type(result.dtype)
            except Exception:
                res_type = 'UNDEFINED'
            row.append(f"    {EX_ORT_TYPES.get(res_type, 'ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED')},\t// {name}({types[i]}, {types[j]}) -> {res_type}\n")
        schema += ("" + "".join(row))
    schema += ("};")
    return template.replace('{TABLE}', schema)

def ex_emit_unary(name, size, types, callable, template):
    schema = f"static const ONNXTensorElementDataType ort_math_promotion_schema_table_{name}[{size}] = {{\n"
    for i, t in enumerate(EX_NUMPY_TYPES):
        try:
            a = np.array([1], dtype=t)
            result = callable(a)
            res_type = ex_get_type(result.dtype)
        except Exception:
            res_type = 'UNDEFINED'
        schema += (f"    {EX_ORT_TYPES.get(res_type, 'ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED')},\t// {name}({types[i]}) -> {res_type}\n")
    schema += ("};")
    return template.replace('{TABLE}', schema)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function', required=False,
        help='Function name (e.g. add, multiply)')
    parser.add_argument('-c', '--code',     required=False,
        help='Code (lambda) to use instead of function')
    parser.add_argument('-b', '--binary',   action='store_true',
        help='Generate binary promotion schema')
    parser.add_argument('-u', '--unary',    action='store_true',
        help='Generate unary promotion schema')
    parser.add_argument('-n', '--name',     required=False,
        help='Schema name (function by default)')
    parser.add_argument('-p', '--preserve', action='store_true', required=False, default=False,
        help='Preserve operands')
    parser.add_argument('-t', '--template', required=False, default='template.h',
        help='Use template file')
    parser.add_argument('-w', '--write',    action='store_true',
        help='Write to schema directory (requires template)')
    args = parser.parse_args()

    if args.unary and args.binary:
        print("Error: Cannot generate both unary and binary schemas.", file=sys.stderr)
        sys.exit(1)

    if not args.unary and not args.binary:
        print("Error: Must specify either --unary or --binary.", file=sys.stderr)
        sys.exit(1)

    if args.function and args.code:
        print("Error: Cannot specify both function and code.", file=sys.stderr)
        sys.exit(1)

    if not args.function and not args.code:
        print("Error: Must specify either --function or --code.", file=sys.stderr)
        sys.exit(1)

    if not args.name:
        if not args.function:
            print("Error: Must specify a name for the schema (where no function is used).", file=sys.stderr)
            sys.exit(1)
        else:
            args.name = args.function

    if args.template:
        if not os.path.exists(args.template):
            args.template = os.path.join(
                os.path.dirname(__file__), 'template.h')
        if not os.path.exists(args.template):
            print(f"Error: Cannot resolve path of template file {args.template}", file=sys.stderr)
            sys.exit(1)

    if args.write and not args.template:
        print("Error: Must specify a template file when writing to schema directory.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(args.template, 'r') as f:
            template = f.read()
    except FileNotFoundError:
        print(f"Error: Template file '{args.template}' not found.", file=sys.stderr)
        sys.exit(1)

    size  = len(EX_NUMPY_TYPES)
    types = [ex_get_type(t) for t in EX_NUMPY_TYPES]
    callable = ex_get_callable(args)

    template = ex_emit_name(args.name, template)
    template = ex_emit_extract(args.name, sys.argv, template)

    np.seterr(divide='ignore')

    if args.binary:
        template = ex_emit_binary(args.name,
            size, types, callable, template)

    if args.unary:
        template = ex_emit_unary(args.name,
            size, types, callable, template)

    template = ex_emit_indices(args.name, size, types, template)
    template = ex_emit_struct(args.name, size,
        'ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY'
            if args.binary else
                'ORT_MATH_PROMOTION_SCHEMA_KIND_UNARY',
        'ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PROMOTE'
            if not args.preserve else
                'ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PRESERVE',
        template)

    if args.write:
        output = os.path.join(
            os.path.dirname(__file__), 
            "..", # dist
            "..", # .
            "src",
            "maths",
            "schema",
            f"{args.name}.h")
        with open(output, 'w') as f:
            f.write(template)
        print(f"OK -> {output}")
    else:
        print(template)

if __name__ == '__main__':
    main()