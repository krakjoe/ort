# Maths Naming Conventions

All math operation implementations (frontend and backend) must strictly adhere to the conventions set out in this document:

# Function signatures:

```c
/* {{{ Binary operation */
void ort_math_frontend_%op%_%type%(
    void *result, 
    const void *a, const void *b, size_t count); /* }}} */

/* {{{ Unary operation */
void ort_math_frontend_%op%_%type%(
    void *result, const void *a, size_t count); /* }}} */

/* {{{ Scalar operation */
void ort_math_frontend_%op%_%type%_scalar(
    void *result,
    const void *a, const void *b, size_t count); /* }}} */

/* {{{ Vectorized operation */
void ort_math_backend_%op%_%type%(
    void *result, const void *a, const void *b, size_t count); /* }}} */

/* {{{ Supported %type% */
enum {
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    float,
    double,
    zend_bool
};
/* }}} */
```

# Local variables:

| Name         | Description                          | Applies to         |
|--------------|--------------------------------------|--------------------|
| va           | a cast to type*                      | All                |
| vb           | b cast to type*                      | Binary             |
| sb           | b cast to type (scalar)              | Scalar             |
| res          | result cast to type*                 | All                |
| count        | Number of elements                   | All                |
| ma, mb, mr   | SIMD register variables              | SIMD               |
| mc           | Number of elements processed by SIMD | SIMD               |
| mw           | SIMD vector width                    | SIMD               |

# Goal

The maths layer contains a lot of macro generations and hand crafted code. By strictly conforming to convention, we make debug and development time as easy as possible.

# Deviation

Deviations are **never** justified for anything; all code **must** conform.