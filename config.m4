dnl config.m4 for extension ort

PHP_ARG_ENABLE([ort],
  [whether to enable ort support],
  [AS_HELP_STRING([--enable-ort], [Enable ort extension])])

PHP_ARG_ENABLE([ort-backend],
  [which backend to use for optimizations],
  [AS_HELP_STRING([--enable-ort-backend@<:@=TYPE@:>@], [Enable backend optimizations. TYPE can be: auto, none, wasm, neon, avx2, sse41, sse2 (default: auto)])],
  [auto],
  [no])

PHP_ARG_ENABLE([ort-pool],
  [whether to enable pooling (threads)],
  [AS_HELP_STRING([--enable-ort-pool], [Enable pooling (threads), default=yes])], yes, no)

PHP_ARG_WITH([ort-onnx],
  [whether to enable ONNX Runtime support],
  [AS_HELP_STRING([--with-ort-onnx], [Enable ONNX Runtime support])], no, no)

AS_VAR_IF([PHP_ORT], [no],, [
  dnl ============================================================
  dnl ISA Extension Detection (Prologue)
  dnl ============================================================
  AC_MSG_CHECKING([for ISA extension support])
  AC_MSG_RESULT([working])
  
  dnl Initialize detection variables
  PHP_ORT_HAS_WASM="no"
  PHP_ORT_HAS_NEON="no" 
  PHP_ORT_HAS_AVX2="no"
  PHP_ORT_HAS_SSE41="no"
  PHP_ORT_HAS_SSE2="no"
  
  dnl Check for WASM SIMD support
  AX_CHECK_COMPILE_FLAG([-msimd128], [
    AC_CHECK_HEADERS([wasm_simd128.h], [
      PHP_ORT_HAS_WASM="yes"
    ], [], [AC_INCLUDES_DEFAULT])
  ], [])

  dnl Check for NEON support (if no wasm)
  if test "$PHP_ORT_HAS_WASM" == "no"; then
    AX_CHECK_COMPILE_FLAG([-march=armv8-a+simd], [
      AC_CHECK_HEADERS([arm_neon.h], [
        PHP_ORT_HAS_NEON="yes"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  dnl Check for AVX2 support (if no wasm/neon)
  if test "$PHP_ORT_HAS_WASM" == "no" &&
     test "$PHP_ORT_HAS_NEON" == "no"; then
    AX_CHECK_COMPILE_FLAG([-mavx2], [
      AC_CHECK_HEADERS([immintrin.h], [
        PHP_ORT_HAS_AVX2="yes"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  if test "$PHP_ORT_HAS_WASM" == "no" &&
     test "$PHP_ORT_HAS_NEON" == "no"; then
    dnl Check for SSE4.1 support (if no wasm/neon)
    AX_CHECK_COMPILE_FLAG([-msse4.1], [
      AC_CHECK_HEADERS([smmintrin.h], [
        PHP_ORT_HAS_SSE41="yes"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  if test "$PHP_ORT_HAS_WASM" == "no" &&
     test "$PHP_ORT_HAS_NEON" == "no"; then
    dnl Check for SSE2 support (if no wasm/neon)
    AX_CHECK_COMPILE_FLAG([-msse2], [
      AC_CHECK_HEADERS([emmintrin.h], [
        PHP_ORT_HAS_SSE2="yes"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  dnl Report detected ISA extensions
  PHP_ORT_ISA_DETECTED=""
  if test "$PHP_ORT_HAS_WASM" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED WASM"
  fi
  if test "$PHP_ORT_HAS_NEON" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED NEON"
  fi
  if test "$PHP_ORT_HAS_AVX2" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED AVX2"
  fi
  if test "$PHP_ORT_HAS_SSE41" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED SSE4.1"
  fi
  if test "$PHP_ORT_HAS_SSE2" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED SSE2"
  fi

  AC_MSG_CHECKING([for detected ISA extensions])
  if test -z "$PHP_ORT_ISA_DETECTED"; then
    AC_MSG_RESULT([none])
  else
    AC_MSG_RESULT([$PHP_ORT_ISA_DETECTED])
  fi

  dnl ============================================================
  dnl ONNX Runtime Support
  dnl ============================================================
  AC_MSG_CHECKING([for onnxruntime support])
  AS_VAR_IF([PHP_ORT_ONNX], [no], [
    AC_MSG_RESULT([no])
  ], [
    dnl Check for ONNX Runtime library using pkg-config
    PKG_CHECK_MODULES([ONNXRUNTIME], [libonnxruntime >= 1.16])

    PHP_EVAL_INCLINE([$ONNXRUNTIME_CFLAGS])
    PHP_EVAL_LIBLINE([$ONNXRUNTIME_LIBS], [ORT_SHARED_LIBADD])
    AC_DEFINE(HAVE_ONNXRUNTIME, 1,
      [Defined to 1 where we should link against onnxruntime])
    AC_MSG_RESULT([$ONNXRUNTIME_LIBS])
  ])

  dnl ============================================================
  dnl Atomic Operations Support
  dnl ============================================================
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

  dnl ============================================================
  dnl Source File Definitions
  dnl ============================================================
  PHP_ORT_SRC_DIR="src"

  PHP_ORT_CORE_IMPL=m4_normalize("
    $PHP_ORT_SRC_DIR/ort.c
    $PHP_ORT_SRC_DIR/env.c
    $PHP_ORT_SRC_DIR/tensor.c
    $PHP_ORT_SRC_DIR/status.c
    $PHP_ORT_SRC_DIR/maths.c
    $PHP_ORT_SRC_DIR/alloc.c
    $PHP_ORT_SRC_DIR/generators.c
    $PHP_ORT_SRC_DIR/generators/random.c
  ")

  AS_VAR_IF([PHP_ORT_ONNX], [no], [], [
    PHP_ORT_CORE_IMPL=m4_normalize("
      $PHP_ORT_CORE_IMPL
      $PHP_ORT_SRC_DIR/options.c
      $PHP_ORT_SRC_DIR/runtime.c
      $PHP_ORT_SRC_DIR/model.c
      $PHP_ORT_SRC_DIR/iterator.c
    ")
  ])

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
    $PHP_ORT_MATHS_FRONTEND_DIR/softmax.c
    $PHP_ORT_MATHS_FRONTEND_DIR/argmax.c
  ")
  PHP_ORT_MATHS_IMPL=m4_normalize("
    $PHP_ORT_MATHS_DIR/broadcast.c
    $PHP_ORT_MATHS_DIR/cast.c
    $PHP_ORT_MATHS_DIR/core.c
    $PHP_ORT_MATHS_DIR/dispatch.c
    $PHP_ORT_MATHS_DIR/promotion.c
    $PHP_ORT_MATHS_DIR/result.c
    $PHP_ORT_MATHS_DIR/validate.c
    $PHP_ORT_MATHS_DIR/pool.c
    $PHP_ORT_MATHS_FRONTEND_IMPL
  ")

  dnl ============================================================
  dnl Backend Configuration
  dnl ============================================================
  AC_MSG_CHECKING([for backend configuration])
  AS_VAR_IF([PHP_ORT_BACKEND], [no], [
    AC_MSG_RESULT([disabled])
  ], [
    PHP_ORT_BACKEND_LEVEL="none"
    PHP_ORT_BACKEND_DIR="$PHP_ORT_MATHS_DIR/backend"
    PHP_ORT_BACKEND_CFLAGS=""
    PHP_ORT_BACKEND_IMPL=""

    dnl Process explicit backend choice
    AS_CASE([$PHP_ORT_BACKEND],
      [auto], [],
      [none], [PHP_ORT_BACKEND="no"],
      [wasm], [PHP_ORT_WASM="yes"],
      [neon], [PHP_ORT_NEON="yes"],
      [avx2], [PHP_ORT_AVX2="yes"],
      [sse41], [PHP_ORT_SSE41="yes"],
      [sse2], [PHP_ORT_SSE2="yes"],
      [AC_MSG_ERROR([Invalid backend TYPE: $PHP_ORT_BACKEND. Valid options: auto, none, wasm, neon, avx2, sse41, sse2])])

    dnl Check for conflicting backend flags
    PHP_ORT_BACKEND_COUNT=0
    if test "x$PHP_ORT_WASM" = "xyes"; then
      PHP_ORT_BACKEND_COUNT=`expr $PHP_ORT_BACKEND_COUNT + 1`
    fi
    if test "x$PHP_ORT_NEON" = "xyes"; then
      PHP_ORT_BACKEND_COUNT=`expr $PHP_ORT_BACKEND_COUNT + 1`
    fi
    if test "x$PHP_ORT_AVX2" = "xyes"; then
      PHP_ORT_BACKEND_COUNT=`expr $PHP_ORT_BACKEND_COUNT + 1`
    fi
    if test "x$PHP_ORT_SSE41" = "xyes"; then
      PHP_ORT_BACKEND_COUNT=`expr $PHP_ORT_BACKEND_COUNT + 1`
    fi
    if test "x$PHP_ORT_SSE2" = "xyes"; then
      PHP_ORT_BACKEND_COUNT=`expr $PHP_ORT_BACKEND_COUNT + 1`
    fi

    if test $PHP_ORT_BACKEND_COUNT -gt 1; then
      AC_MSG_ERROR([Only one backend can be enabled at a time (found $PHP_ORT_BACKEND_COUNT enabled backends)])
    fi

    dnl If no specific backend was requested, auto-detect in priority order
    if test $PHP_ORT_BACKEND_COUNT -eq 0; then
      AC_MSG_RESULT([auto-detecting])
      
      dnl Check which backends were explicitly disabled by looking at enable_ort_* variables
      dnl that autoconf sets when processing --disable-ort-* flags
      
      dnl Select best available backend in priority order (respecting explicit --disable flags)
      if test "$PHP_ORT_HAS_WASM" = "yes" && test "$enable_ort_wasm" != "no"; then
        PHP_ORT_WASM="yes"
      elif test "$PHP_ORT_HAS_NEON" = "yes" && test "$enable_ort_neon" != "no"; then
        PHP_ORT_NEON="yes"
      elif test "$PHP_ORT_HAS_AVX2" = "yes" && test "$enable_ort_avx2" != "no"; then
        PHP_ORT_AVX2="yes"
      elif test "$PHP_ORT_HAS_SSE41" = "yes" && test "$enable_ort_sse41" != "no"; then
        PHP_ORT_SSE41="yes"
      elif test "$PHP_ORT_HAS_SSE2" = "yes" && test "$enable_ort_sse2" != "no"; then
        PHP_ORT_SSE2="yes"
      else
        PHP_ORT_BACKEND="no"
        AC_MSG_RESULT([no suitable backend found, disabling])
      fi
    else
      AC_MSG_RESULT([using requested backend])
    fi

    dnl Configure the selected backend using simple variable checks
    if test "$PHP_ORT_BACKEND" != "no"; then
      dnl WASM backend
      if test "$PHP_ORT_WASM" = "yes"; then
        if test "$PHP_ORT_HAS_WASM" != "yes"; then
          AC_MSG_ERROR([WASM backend requested but not available])
        fi
        if test "$abs_srcdir" != "$abs_builddir"; then
          AC_MSG_ERROR([WASM backend only supports in-tree (source == build dir) builds])
        fi
        PHP_ORT_BACKEND_CFLAGS="-msimd128"
        PHP_ORT_BACKEND_LEVEL="WASM"
        PHP_ORT_BACKEND_IMPL=m4_normalize("
          $PHP_ORT_BACKEND_DIR/wasm/abs.c
          $PHP_ORT_BACKEND_DIR/wasm/add.c
          $PHP_ORT_BACKEND_DIR/wasm/ceil.c
          $PHP_ORT_BACKEND_DIR/wasm/div.c
          $PHP_ORT_BACKEND_DIR/wasm/floor.c
          $PHP_ORT_BACKEND_DIR/wasm/matmul.c
          $PHP_ORT_BACKEND_DIR/wasm/mul.c
          $PHP_ORT_BACKEND_DIR/wasm/neg.c
          $PHP_ORT_BACKEND_DIR/wasm/recip.c
          $PHP_ORT_BACKEND_DIR/wasm/round.c
          $PHP_ORT_BACKEND_DIR/wasm/sign.c
          $PHP_ORT_BACKEND_DIR/wasm/sqrt.c
          $PHP_ORT_BACKEND_DIR/wasm/sub.c
          $PHP_ORT_BACKEND_DIR/wasm/trunc.c
          $PHP_ORT_BACKEND_DIR/wasm/impl.c
        ")
        PHP_ORT_POOL="no"
        ext_shared=no
      fi

      dnl NEON backend
      if test "$PHP_ORT_NEON" = "yes"; then
        if test "$PHP_ORT_HAS_NEON" != "yes"; then
          AC_MSG_ERROR([NEON backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="-march=armv8-a+simd"
        PHP_ORT_BACKEND_LEVEL="NEON"
        PHP_ORT_BACKEND_IMPL=m4_normalize("
          $PHP_ORT_BACKEND_DIR/neon/abs.c
          $PHP_ORT_BACKEND_DIR/neon/add.c
          $PHP_ORT_BACKEND_DIR/neon/ceil.c
          $PHP_ORT_BACKEND_DIR/neon/div.c
          $PHP_ORT_BACKEND_DIR/neon/floor.c
          $PHP_ORT_BACKEND_DIR/neon/matmul.c
          $PHP_ORT_BACKEND_DIR/neon/neg.c
          $PHP_ORT_BACKEND_DIR/neon/recip.c
          $PHP_ORT_BACKEND_DIR/neon/round.c
          $PHP_ORT_BACKEND_DIR/neon/sign.c
          $PHP_ORT_BACKEND_DIR/neon/sub.c
          $PHP_ORT_BACKEND_DIR/neon/sqrt.c
          $PHP_ORT_BACKEND_DIR/neon/trunc.c
          $PHP_ORT_BACKEND_DIR/neon/mul.c
          $PHP_ORT_BACKEND_DIR/neon/impl.c
        ")
      fi

      dnl AVX2 backend
      if test "$PHP_ORT_AVX2" = "yes"; then
        if test "$PHP_ORT_HAS_AVX2" != "yes"; then
          AC_MSG_ERROR([AVX2 backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="-mavx2"
        PHP_ORT_BACKEND_LEVEL="AVX2"
        PHP_ORT_BACKEND_IMPL=m4_normalize("
          $PHP_ORT_BACKEND_DIR/avx2/add.c
          $PHP_ORT_BACKEND_DIR/avx2/sub.c
          $PHP_ORT_BACKEND_DIR/avx2/dot.c
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
      fi

      dnl SSE4.1 backend
      if test "$PHP_ORT_SSE41" = "yes"; then
        if test "$PHP_ORT_HAS_SSE41" != "yes"; then
          AC_MSG_ERROR([SSE4.1 backend requested but not available])
        fi
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
      fi

      dnl SSE2 backend
      if test "$PHP_ORT_SSE2" = "yes"; then
        if test "$PHP_ORT_HAS_SSE2" != "yes"; then
          AC_MSG_ERROR([SSE2 backend requested but not available])
        fi
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
      fi

      dnl Set backend defines and report result
      AC_MSG_CHECKING([for backend build])
      if test "$PHP_ORT_BACKEND_LEVEL" != "none"; then
        AC_DEFINE(ORT_BACKEND_ENABLED, 1, [ort backend optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_NAME, "$PHP_ORT_BACKEND_LEVEL", [ort backend name])
        PHP_SUBST(PHP_ORT_BACKEND_CFLAGS)
        AC_MSG_RESULT([$PHP_ORT_BACKEND_LEVEL with $PHP_ORT_BACKEND_CFLAGS])
      else
        AC_MSG_RESULT([none])
      fi
    fi
  ])

  dnl ============================================================
  dnl Pooling Support
  dnl ============================================================
  AC_MSG_CHECKING([for pooling support])
  AS_VAR_IF([PHP_ORT_POOL], [no], [
    AC_MSG_RESULT([no])
  ], [
    AC_DEFINE(HAVE_ORT_POOL, 1,
      [Defined to 1 where we should use pooling (threads)])
    AC_MSG_RESULT([enabled])
  ])

  dnl ============================================================
  dnl Extension Build Configuration
  dnl ============================================================
  dnl Add source files
  PHP_NEW_EXTENSION(ort, [php_ort.c $PHP_ORT_CORE_IMPL $PHP_ORT_MATHS_IMPL $PHP_ORT_BACKEND_IMPL], $ext_shared,, [${PHP_ORT_BACKEND_CFLAGS}])

  dnl Add include paths
  PHP_ADD_INCLUDE([$ext_srcdir/src])

  PHP_ADD_INCLUDE([$ext_builddir/$PHP_ORT_MATHS_DIR])
  PHP_ADD_BUILD_DIR([$ext_builddir/$PHP_ORT_MATHS_DIR])

  if test "$PHP_ORT_BACKEND_LEVEL" != "none"; then
    PHP_ADD_INCLUDE([$ext_builddir/$PHP_ORT_BACKEND_DIR])
    PHP_ADD_BUILD_DIR([$ext_builddir/$PHP_ORT_BACKEND_DIR])
  fi
  
  PHP_SUBST([ORT_SHARED_LIBADD])
  PHP_ADD_MAKEFILE_FRAGMENT(Makefile.frag)
])