dnl config.m4 for extension ort

PHP_ARG_ENABLE([ort],
  [whether to enable ort support],
  [AS_HELP_STRING([--enable-ort], [Enable ort extension])])

PHP_ARG_ENABLE([ort-coverage],
  [whether to enable ort coverage],
  [AS_HELP_STRING([--enable-ort-coverage], [Enable ort coverage support, default=no])], no, no)

PHP_ARG_ENABLE([ort-simd],
  [whether to enable SIMD optimizations],
  [AS_HELP_STRING([--enable-ort-simd], [Enable SIMD optimizations, default=yes])], yes, no)

PHP_ARG_ENABLE([ort-sse2],
  [whether to enable SSE2 support],
  [AS_HELP_STRING([--enable-ort-sse2], [Enable SSE2 support])], yes, no)

PHP_ARG_ENABLE([ort-sse41],
  [whether to enable SSE4.1 support],
  [AS_HELP_STRING([--enable-ort-sse41], [Enable SSE4.1 support])], yes, no)

PHP_ARG_ENABLE([ort-avx2],
  [whether to enable AVX2 support],
  [AS_HELP_STRING([--enable-ort-avx2], [Enable AVX2 support])], yes, no)

AS_VAR_IF([PHP_ORT], [no],, [
  dnl Check for ONNX Runtime library using pkg-config
  PKG_CHECK_MODULES([ONNX], [libonnxruntime >= 1.16])

  PHP_EVAL_INCLINE([$ONNX_CFLAGS])
  PHP_EVAL_LIBLINE([$ONNX_LIBS], [ORT_SHARED_LIBADD])

  AC_MSG_CHECKING([for __atomic_add_fetch])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([], [[
    int variable = 1;
    return (__atomic_add_fetch(&variable, 1, __ATOMIC_SEQ_CST)) ? 1 : 0;
  ]])], [
    AC_MSG_RESULT([yes])
    AC_DEFINE(HAVE_BUILTIN_ATOMIC_CPP11, 1, [Define to 1 if supports __atomic_add_fetch()])
  ], [
    AC_MSG_RESULT([no])
  ])

  dnl Check for coverage support
  AC_MSG_CHECKING([for coverage support])
  AS_VAR_IF([PHP_ORT_COVERAGE], [no], [
        AC_MSG_RESULT([no])
  ], [
    PHP_ADD_MAKEFILE_FRAGMENT(Makefile.frag)
    AC_MSG_RESULT([yes])
  ])

  PHP_ORT_SRC_DIR="src"

  PHP_ORT_CORE_IMPL=m4_normalize("
    $PHP_ORT_SRC_DIR/ort.c
    $PHP_ORT_SRC_DIR/env.c
    $PHP_ORT_SRC_DIR/runtime.c
    $PHP_ORT_SRC_DIR/model.c
    $PHP_ORT_SRC_DIR/tensor.c
    $PHP_ORT_SRC_DIR/status.c
    $PHP_ORT_SRC_DIR/options.c
    $PHP_ORT_SRC_DIR/iterator.c
    $PHP_ORT_SRC_DIR/maths.c
  ")

  PHP_ORT_MATHS_DIR=$PHP_ORT_SRC_DIR/maths
  PHP_ORT_MATHS_FRONTEND_DIR=$PHP_ORT_MATHS_DIR/frontend
  PHP_ORT_MATHS_FRONTEND_IMPL=m4_normalize("
    $PHP_ORT_MATHS_FRONTEND_DIR/add.c
    $PHP_ORT_MATHS_FRONTEND_DIR/div.c
    $PHP_ORT_MATHS_FRONTEND_DIR/neg.c
    $PHP_ORT_MATHS_FRONTEND_DIR/mul.c
    $PHP_ORT_MATHS_FRONTEND_DIR/recip.c
    $PHP_ORT_MATHS_FRONTEND_DIR/sign.c
    $PHP_ORT_MATHS_FRONTEND_DIR/sub.c
    $PHP_ORT_MATHS_FRONTEND_DIR/mod.c
    $PHP_ORT_MATHS_FRONTEND_DIR/pow.c
    $PHP_ORT_MATHS_FRONTEND_DIR/sqrt.c
    $PHP_ORT_MATHS_FRONTEND_DIR/matmul.c
    $PHP_ORT_MATHS_FRONTEND_DIR/sum.c
    $PHP_ORT_MATHS_FRONTEND_DIR/mean.c
    $PHP_ORT_MATHS_FRONTEND_DIR/min.c
    $PHP_ORT_MATHS_FRONTEND_DIR/max.c
    $PHP_ORT_MATHS_FRONTEND_DIR/dot.c
    $PHP_ORT_MATHS_FRONTEND_DIR/real.c
  ")
  PHP_ORT_MATHS_IMPL=m4_normalize("
    $PHP_ORT_MATHS_DIR/broadcast.c
    $PHP_ORT_MATHS_DIR/cast.c
    $PHP_ORT_MATHS_DIR/core.c
    $PHP_ORT_MATHS_DIR/dispatch.c
    $PHP_ORT_MATHS_DIR/promotion.c
    $PHP_ORT_MATHS_DIR/result.c
    $PHP_ORT_MATHS_DIR/validate.c
    $PHP_ORT_MATHS_FRONTEND_IMPL
  ")

  dnl SIMD feature detection
  AC_MSG_CHECKING([for SIMD configuration])
  AS_VAR_IF([PHP_ORT_SIMD], [no], [
    AC_MSG_RESULT([disabled])
  ], [
    AC_MSG_RESULT([enabled])
    PHP_ORT_BACKEND_LEVEL="none"
    PHP_ORT_BACKEND_DIR="$PHP_ORT_MATHS_DIR/backend"

    dnl Check for SSE2 support (lowest level)
    if test "$PHP_ORT_SSE2" != "no"; then
      AX_CHECK_COMPILE_FLAG([-msse2], [
        AC_CHECK_HEADERS([emmintrin.h], [
          PHP_ORT_BACKEND_CFLAGS="-msse2"
          PHP_ORT_BACKEND_LEVEL="SSE2"
          PHP_ORT_BACKEND_IMPL=m4_normalize("
            $PHP_ORT_BACKEND_DIR/sse2/abs.c
            $PHP_ORT_BACKEND_DIR/sse2/add.c
            $PHP_ORT_BACKEND_DIR/sse2/div.c
            $PHP_ORT_BACKEND_DIR/sse2/matmul.c
            $PHP_ORT_BACKEND_DIR/sse2/mul.c
            $PHP_ORT_BACKEND_DIR/sse2/neg.c
            $PHP_ORT_BACKEND_DIR/sse2/recip.c
            $PHP_ORT_BACKEND_DIR/sse2/sign.c
            $PHP_ORT_BACKEND_DIR/sse2/sqrt.c
            $PHP_ORT_BACKEND_DIR/sse2/sub.c
            $PHP_ORT_BACKEND_DIR/sse2/impl.c
          ")
        ], [
          if test "$PHP_ORT_SSE2" = "yes"; then
            AC_MSG_ERROR([SSE2 headers not found])
          fi
        ])
      ], [
        if test "$PHP_ORT_SSE2" = "yes"; then
          AC_MSG_ERROR([SSE2 support requested but not available])
        fi
      ])
    fi

    dnl Check for SSE4.1 support (higher level)
    if test "$PHP_ORT_SSE41" != "no"; then
      AX_CHECK_COMPILE_FLAG([-msse4.1], [
        AC_CHECK_HEADERS([smmintrin.h], [
          PHP_ORT_BACKEND_CFLAGS="-msse4.1"
          PHP_ORT_BACKEND_LEVEL="SSE4.1"
          PHP_ORT_BACKEND_IMPL=m4_normalize("
            $PHP_ORT_BACKEND_DIR/sse41/add.c
            $PHP_ORT_BACKEND_DIR/sse41/sub.c
            $PHP_ORT_BACKEND_DIR/sse41/matmul.c
            $PHP_ORT_BACKEND_DIR/sse41/mul.c
            $PHP_ORT_BACKEND_DIR/sse41/div.c
            $PHP_ORT_BACKEND_DIR/sse41/sqrt.c
            $PHP_ORT_BACKEND_DIR/sse41/neg.c
            $PHP_ORT_BACKEND_DIR/sse41/ceil.c
            $PHP_ORT_BACKEND_DIR/sse41/floor.c
            $PHP_ORT_BACKEND_DIR/sse41/round.c
            $PHP_ORT_BACKEND_DIR/sse41/abs.c
            $PHP_ORT_BACKEND_DIR/sse41/sign.c
            $PHP_ORT_BACKEND_DIR/sse41/recip.c
            $PHP_ORT_BACKEND_DIR/sse41/trunc.c
            $PHP_ORT_BACKEND_DIR/sse41/impl.c
          ")
        ], [
          if test "$PHP_ORT_SSE41" = "yes"; then
            AC_MSG_ERROR([SSE4.1 headers not found])
          fi
        ])
      ], [
        if test "$PHP_ORT_SSE41" = "yes"; then
          AC_MSG_ERROR([SSE4.1 support requested but not available])
        fi
      ])
    fi

    dnl Check for AVX2 support (highest level)
    if test "$PHP_ORT_AVX2" != "no"; then
      AX_CHECK_COMPILE_FLAG([-mavx2], [
        AC_CHECK_HEADERS([immintrin.h], [
          PHP_ORT_BACKEND_CFLAGS="-mavx2"
          PHP_ORT_BACKEND_LEVEL="AVX2"
          PHP_ORT_BACKEND_IMPL=m4_normalize("
            $PHP_ORT_BACKEND_DIR/avx2/add.c
            $PHP_ORT_BACKEND_DIR/avx2/sub.c
            $PHP_ORT_BACKEND_DIR/avx2/matmul.c
            $PHP_ORT_BACKEND_DIR/avx2/mul.c
            $PHP_ORT_BACKEND_DIR/avx2/div.c
            $PHP_ORT_BACKEND_DIR/avx2/sqrt.c
            $PHP_ORT_BACKEND_DIR/avx2/neg.c
            $PHP_ORT_BACKEND_DIR/avx2/ceil.c
            $PHP_ORT_BACKEND_DIR/avx2/floor.c
            $PHP_ORT_BACKEND_DIR/avx2/round.c
            $PHP_ORT_BACKEND_DIR/avx2/abs.c
            $PHP_ORT_BACKEND_DIR/avx2/sign.c
            $PHP_ORT_BACKEND_DIR/avx2/recip.c
            $PHP_ORT_BACKEND_DIR/avx2/trunc.c
            $PHP_ORT_BACKEND_DIR/avx2/impl.c
          ")
        ], [
          if test "$PHP_ORT_AVX2" = "yes"; then
            AC_MSG_ERROR([AVX2 headers not found])
          fi
        ])
      ], [
        if test "$PHP_ORT_AVX2" = "yes"; then
          AC_MSG_ERROR([AVX2 support requested but not available])
        fi
      ])
    fi

    dnl Define the highest available SIMD level
    AC_MSG_CHECKING([for SIMD build $PHP_SIMD_IMPL])
    if test "$PHP_ORT_BACKEND_LEVEL" = "AVX2"; then
      AC_DEFINE(HAVE_AVX2, 1, [AVX2 support available])
      AC_DEFINE(ORT_SIMD_ENABLED, 1, [SIMD optimizations enabled])
      PHP_SUBST(PHP_ORT_BACKEND_CFLAGS)
      AC_MSG_RESULT([$PHP_ORT_BACKEND_LEVEL with $PHP_ORT_BACKEND_CFLAGS])
    elif test "$PHP_ORT_BACKEND_LEVEL" = "SSE4.1"; then
      AC_DEFINE(HAVE_SSE41, 1, [SSE4.1 support available])
      AC_DEFINE(ORT_SIMD_ENABLED, 1, [SIMD optimizations enabled])
      PHP_SUBST(PHP_ORT_BACKEND_CFLAGS)
      AC_MSG_RESULT([$PHP_ORT_BACKEND_LEVEL with $PHP_ORT_BACKEND_CFLAGS])
    elif test "$PHP_ORT_BACKEND_LEVEL" = "SSE2"; then
      AC_DEFINE(HAVE_SSE2, 1, [SSE2 support available])
      AC_DEFINE(ORT_SIMD_ENABLED, 1, [SIMD optimizations enabled])
      PHP_SUBST(PHP_ORT_BACKEND_CFLAGS)
      AC_MSG_RESULT([$PHP_ORT_BACKEND_LEVEL with $PHP_ORT_BACKEND_CFLAGS])
    else
      AC_MSG_RESULT([none])
    fi
  ])

  dnl Add source files
  PHP_NEW_EXTENSION(ort, [php_ort.c $PHP_ORT_CORE_IMPL $PHP_ORT_MATHS_IMPL $PHP_ORT_BACKEND_IMPL], $ext_shared,, [${PHP_ORT_BACKEND_CFLAGS}])

  dnl Add include paths
  PHP_ADD_INCLUDE([$ext_srcdir/src])

  PHP_ADD_INCLUDE([
    $ext_builddir/$PHP_ORT_MATHS_DIR])
  PHP_ADD_BUILD_DIR([
    $ext_builddir/$PHP_ORT_MATHS_DIR])

  if test "$PHP_ORT_BACKEND_LEVEL" != "none"; then
    PHP_ADD_INCLUDE([
      $ext_builddir/$PHP_ORT_BACKEND_DIR
    ])
    PHP_ADD_BUILD_DIR([
      $ext_builddir/$PHP_ORT_BACKEND_DIR
    ])
  fi

  PHP_SUBST([ORT_SHARED_LIBADD])
])
