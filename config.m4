dnl config.m4 for extension ort

PHP_ARG_ENABLE([ort],
  [whether to enable ort support],
  [AS_HELP_STRING([--enable-ort], [Enable ort extension])])

PHP_ARG_ENABLE([ort-backend],
  [which backend(s) to use for optimizations],
  [AS_HELP_STRING([--enable-ort-backend@<:@=TYPE@:>@], [Enable backend optimizations. TYPE can be: auto, none, or comma-separated list like cuda,auto (default: auto)])],
  [auto],
  [no])

PHP_ARG_ENABLE([ort-pool],
  [whether to enable pooling (threads)],
  [AS_HELP_STRING([--enable-ort-pool], [Enable pooling (threads), default=yes])], yes, no)

PHP_ARG_WITH([ort-onnx],
  [whether to enable ONNX Runtime support],
  [AS_HELP_STRING([--with-ort-onnx], [Enable ONNX Runtime support])], no, no)

PHP_ARG_ENABLE([ort-coverage],
  [whether to enable coverage],
  [AS_HELP_STRING([--enable-ort-coverage], [Enable coverage, default=no])], no, no)

AS_VAR_IF([PHP_ORT], [no],, [
  dnl ============================================================
  dnl ISA Extension Detection (Prologue)
  dnl ============================================================
  AC_MSG_CHECKING([for ISA extension support])
  AC_MSG_RESULT([working])
  
  dnl Initialize detection variables
  PHP_ORT_HAS_CUDA="no"
  PHP_ORT_HAS_WASM="no"
  PHP_ORT_HAS_NEON="no"
  PHP_ORT_HAS_RISCV64="no"
  PHP_ORT_HAS_AVX512="no"
  PHP_ORT_HAS_AVX2="no"
  PHP_ORT_HAS_SSE41="no"
  PHP_ORT_HAS_SSE2="no"
  
  dnl Check for CUDA support
  dnl First check environment variables
  if test -n "$CUDA_HOME"; then
    CUDA_DIR="$CUDA_HOME"
  elif test -n "$CUDA_PATH"; then
    CUDA_DIR="$CUDA_PATH"
  else
    dnl Try to find nvcc in PATH
    AC_PATH_PROG([NVCC], [nvcc], [no])
    if test "$NVCC" != "no"; then
      dnl Derive CUDA directory from nvcc location
      CUDA_DIR=$(dirname $(dirname $NVCC))
      PHP_SUBST(NVCC)
    else
      dnl Try standard installation paths
      for dir in /usr/local/cuda /opt/cuda /usr/cuda; do
        if test -d "$dir"; then
          CUDA_DIR="$dir"
          break
        fi
      done
    fi
  fi

  dnl If we found a CUDA directory, verify it has what we need
  if test -n "$CUDA_DIR"; then
    save_CPPFLAGS="$CPPFLAGS"
    save_LDFLAGS="$LDFLAGS"
    
    CPPFLAGS="$CPPFLAGS -I$CUDA_DIR/include"
    LDFLAGS="$LDFLAGS -L$CUDA_DIR/lib64 -L$CUDA_DIR/lib"
    
    AC_CHECK_HEADERS([cuda_runtime.h], [
      AC_CHECK_LIB([cudart], [cudaMalloc], [
        PHP_ORT_HAS_CUDA="yes"
      ])
    ], [], [AC_INCLUDES_DEFAULT])
    
    CPPFLAGS="$save_CPPFLAGS"
    LDFLAGS="$save_LDFLAGS"
  fi

  dnl Check for WASM SIMD support
  AX_CHECK_COMPILE_FLAG([-msimd128], [
    AC_CHECK_HEADERS([wasm_simd128.h], [
      PHP_ORT_HAS_WASM="yes"
    ], [], [AC_INCLUDES_DEFAULT])
  ], [])

  dnl Check for NEON support
  if test "$PHP_ORT_HAS_WASM" = "no"; then
    AX_CHECK_COMPILE_FLAG([-march=armv8-a+simd], [
      AC_CHECK_HEADERS([arm_neon.h], [
        PHP_ORT_HAS_NEON="yes"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  dnl Check for RISCV64 Vector support
  if test "$PHP_ORT_HAS_WASM" = "no" &&
     test "$PHP_ORT_HAS_NEON" = "no"; then
    AX_CHECK_COMPILE_FLAG([-march=rv64gcv], [
      save_CFLAGS="$CFLAGS"
      CFLAGS="$CFLAGS -march=rv64gcv"
      AC_CHECK_HEADERS([riscv_vector.h], [
        PHP_ORT_HAS_RISCV64="yes"
      ], [], [AC_INCLUDES_DEFAULT])
      CFLAGS="$save_CFLAGS"
    ], [])
  fi

  dnl Check for AVX512 support
  if test "$PHP_ORT_HAS_WASM"    = "no" &&
     test "$PHP_ORT_HAS_NEON"    = "no" &&
     test "$PHP_ORT_HAS_RISCV64" = "no"; then
    AX_CHECK_COMPILE_FLAG([-mavx512f -mavx512bw -mavx512dq -mavx512vl -mevex512], [
      AC_CHECK_HEADERS([immintrin.h], [
        dnl Runtime test for AVX512 instruction support
        AC_MSG_CHECKING([for runtime AVX512 support])
        saved_CFLAGS="$CFLAGS"
        CFLAGS="$CFLAGS -mavx512f -mavx512bw -mavx512dq -mavx512vl -mevex512"
        AC_RUN_IFELSE([AC_LANG_PROGRAM([[
          #include <immintrin.h>
          #include <signal.h>
          #include <setjmp.h>
          static jmp_buf jmp_env;
          static void sigill_handler(int sig) { longjmp(jmp_env, 1); }
        ]], [[
          signal(SIGILL, sigill_handler);
          if (setjmp(jmp_env) == 0) {
            __m512i a = _mm512_setzero_si512();
            __m512i b = _mm512_set1_epi32(1);
            __m512i c = _mm512_add_epi32(a, b);
            volatile int result = _mm512_extract_epi32(c, 0);
            return (result == 1) ? 0 : 1;
          }
          return 1;
        ]])], [
          AC_MSG_RESULT([yes])
          PHP_ORT_HAS_AVX512="yes"
        ], [
          AC_MSG_RESULT([no])
          PHP_ORT_HAS_AVX512="no"
        ], [
          dnl Cross-compilation: assume available if compiler supports it
          AC_MSG_RESULT([assuming yes (cross-compiling)])
          PHP_ORT_HAS_AVX512="yes"
        ])
        CFLAGS="$saved_CFLAGS"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  dnl Check for AVX2 support
  if test "$PHP_ORT_HAS_WASM"    = "no" &&
     test "$PHP_ORT_HAS_NEON"    = "no" &&
     test "$PHP_ORT_HAS_RISCV64" = "no"; then
    AX_CHECK_COMPILE_FLAG([-mavx2], [
      AC_CHECK_HEADERS([immintrin.h], [
        dnl Runtime test for AVX2 instruction support
        AC_MSG_CHECKING([for runtime AVX2 support])
        saved_CFLAGS="$CFLAGS"
        CFLAGS="$CFLAGS -mavx2"
        AC_RUN_IFELSE([AC_LANG_PROGRAM([[
          #include <immintrin.h>
          #include <signal.h>
          #include <setjmp.h>
          static jmp_buf jmp_env;
          static void sigill_handler(int sig) { longjmp(jmp_env, 1); }
        ]], [[
          signal(SIGILL, sigill_handler);
          if (setjmp(jmp_env) == 0) {
            __m256i a = _mm256_setzero_si256();
            __m256i b = _mm256_set1_epi32(1);
            __m256i c = _mm256_add_epi32(a, b);
            volatile int result = _mm256_extract_epi32(c, 0);
            return (result == 1) ? 0 : 1;
          }
          return 1;
        ]])], [
          AC_MSG_RESULT([yes])
          PHP_ORT_HAS_AVX2="yes"
        ], [
          AC_MSG_RESULT([no])
          PHP_ORT_HAS_AVX2="no"
        ], [
          dnl Cross-compilation: assume available if compiler supports it
          AC_MSG_RESULT([assuming yes (cross-compiling)])
          PHP_ORT_HAS_AVX2="yes"
        ])
        CFLAGS="$saved_CFLAGS"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  dnl Check for SSE4.1 support
  if test "$PHP_ORT_HAS_WASM"    = "no" &&
     test "$PHP_ORT_HAS_NEON"    = "no" &&
     test "$PHP_ORT_HAS_RISCV64" = "no"; then
    AX_CHECK_COMPILE_FLAG([-msse4.1], [
      AC_CHECK_HEADERS([smmintrin.h], [
        dnl Runtime test for SSE4.1 instruction support
        AC_MSG_CHECKING([for runtime SSE4.1 support])
        saved_CFLAGS="$CFLAGS"
        CFLAGS="$CFLAGS -msse4.1"
        AC_RUN_IFELSE([AC_LANG_PROGRAM([[
          #include <smmintrin.h>
          #include <signal.h>
          #include <setjmp.h>
          static jmp_buf jmp_env;
          static void sigill_handler(int sig) { longjmp(jmp_env, 1); }
        ]], [[
          signal(SIGILL, sigill_handler);
          if (setjmp(jmp_env) == 0) {
            __m128i a = _mm_setzero_si128();
            __m128i b = _mm_set1_epi32(1);
            __m128i c = _mm_add_epi32(a, b);
            volatile int result = _mm_extract_epi32(c, 0);
            return (result == 1) ? 0 : 1;
          }
          return 1;
        ]])], [
          AC_MSG_RESULT([yes])
          PHP_ORT_HAS_SSE41="yes"
        ], [
          AC_MSG_RESULT([no])
          PHP_ORT_HAS_SSE41="no"
        ], [
          dnl Cross-compilation: assume available if compiler supports it
          AC_MSG_RESULT([assuming yes (cross-compiling)])
          PHP_ORT_HAS_SSE41="yes"
        ])
        CFLAGS="$saved_CFLAGS"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  dnl Check for SSE2 support
  if test "$PHP_ORT_HAS_WASM"    = "no" &&
     test "$PHP_ORT_HAS_NEON"    = "no" &&
     test "$PHP_ORT_HAS_RISCV64" = "no"; then
    AX_CHECK_COMPILE_FLAG([-msse2], [
      AC_CHECK_HEADERS([emmintrin.h], [
        dnl Runtime test for SSE2 instruction support
        AC_MSG_CHECKING([for runtime SSE2 support])
        saved_CFLAGS="$CFLAGS"
        CFLAGS="$CFLAGS -msse2"
        AC_RUN_IFELSE([AC_LANG_PROGRAM([[
          #include <emmintrin.h>
          #include <signal.h>
          #include <setjmp.h>
          static jmp_buf jmp_env;
          static void sigill_handler(int sig) { longjmp(jmp_env, 1); }
        ]], [[
          signal(SIGILL, sigill_handler);
          if (setjmp(jmp_env) == 0) {
            __m128i a = _mm_setzero_si128();
            __m128i b = _mm_set1_epi32(1);
            __m128i c = _mm_add_epi32(a, b);
            volatile int result = _mm_cvtsi128_si32(c);
            return (result == 1) ? 0 : 1;
          }
          return 1;
        ]])], [
          AC_MSG_RESULT([yes])
          PHP_ORT_HAS_SSE2="yes"
        ], [
          AC_MSG_RESULT([no])
          PHP_ORT_HAS_SSE2="no"
        ], [
          dnl Cross-compilation: assume available if compiler supports it
          AC_MSG_RESULT([assuming yes (cross-compiling)])
          PHP_ORT_HAS_SSE2="yes"
        ])
        CFLAGS="$saved_CFLAGS"
      ], [], [AC_INCLUDES_DEFAULT])
    ], [])
  fi

  dnl Report detected ISA extensions
  PHP_ORT_ISA_DETECTED=""
  if test "$PHP_ORT_HAS_CUDA" = "yes"; then
    PHP_ORT_ISA_DETECTED="CUDA"
  fi
  if test "$PHP_ORT_HAS_WASM" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED WASM"
  fi
  if test "$PHP_ORT_HAS_NEON" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED NEON"
  fi
  if test "$PHP_ORT_HAS_RISCV64" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED RISCV64"
  fi
  if test "$PHP_ORT_HAS_AVX512" = "yes"; then
    PHP_ORT_ISA_DETECTED="$PHP_ORT_ISA_DETECTED AVX512"
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
    PHP_ORT_BACKEND_LEVEL=""
    PHP_ORT_BACKEND_DIR="$PHP_ORT_MATHS_DIR/backend"
    PHP_ORT_BACKEND_CFLAGS=""
    PHP_ORT_BACKEND_IMPL=""

    dnl Initialize backend selection flags
    PHP_ORT_USE_CUDA="no"
    PHP_ORT_USE_WASM="no"
    PHP_ORT_USE_NEON="no"
    PHP_ORT_USE_RISCV64="no"
    PHP_ORT_USE_AVX512="no"
    PHP_ORT_USE_AVX2="no"
    PHP_ORT_USE_SSE41="no"
    PHP_ORT_USE_SSE2="no"

    dnl Parse backend specification
    case "$PHP_ORT_BACKEND" in
      auto)
        AC_MSG_RESULT([auto-detecting])
        dnl Auto-select: CUDA if available, plus best CPU backend
        if test "$PHP_ORT_HAS_CUDA" = "yes"; then
          PHP_ORT_USE_CUDA="yes"
        fi
        if test "$PHP_ORT_HAS_WASM" = "yes"; then
          PHP_ORT_USE_WASM="yes"
        elif test "$PHP_ORT_HAS_NEON" = "yes"; then
          PHP_ORT_USE_NEON="yes"
        elif test "$PHP_ORT_HAS_RISCV64" = "yes"; then
          PHP_ORT_USE_RISCV64="yes"
        elif test "$PHP_ORT_HAS_AVX512" = "yes"; then
          PHP_ORT_USE_AVX512="yes"
        elif test "$PHP_ORT_HAS_AVX2" = "yes"; then
          PHP_ORT_USE_AVX2="yes"
        elif test "$PHP_ORT_HAS_SSE41" = "yes"; then
          PHP_ORT_USE_SSE41="yes"
        elif test "$PHP_ORT_HAS_SSE2" = "yes"; then
          PHP_ORT_USE_SSE2="yes"
        fi
        ;;
      none)
        AC_MSG_RESULT([explicitly disabled])
        PHP_ORT_BACKEND="no"
        ;;
      *)
        AC_MSG_RESULT([parsing: $PHP_ORT_BACKEND])
        dnl Parse comma-separated list
        PHP_ORT_CPU_BACKEND_COUNT=0
        saved_IFS="$IFS"
        IFS=','
        for backend in $PHP_ORT_BACKEND; do
          case "$backend" in
            cuda)
              PHP_ORT_USE_CUDA="yes"
              ;;
            auto)
              dnl Auto detect CPU backend
              if test "$PHP_ORT_HAS_WASM" = "yes"; then
                PHP_ORT_USE_WASM="yes"
                PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              elif test "$PHP_ORT_HAS_NEON" = "yes"; then
                PHP_ORT_USE_NEON="yes"
                PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              elif test "$PHP_ORT_HAS_RISCV64" = "yes"; then
                PHP_ORT_USE_RISCV64="yes"
                PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              elif test "$PHP_ORT_HAS_AVX512" = "yes"; then
                PHP_ORT_USE_AVX512="yes"
                PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              elif test "$PHP_ORT_HAS_AVX2" = "yes"; then
                PHP_ORT_USE_AVX2="yes"
                PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              elif test "$PHP_ORT_HAS_SSE41" = "yes"; then
                PHP_ORT_USE_SSE41="yes"
                PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              elif test "$PHP_ORT_HAS_SSE2" = "yes"; then
                PHP_ORT_USE_SSE2="yes"
                PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              fi
              ;;
            wasm)
              PHP_ORT_USE_WASM="yes"
              PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              ;;
            neon)
              PHP_ORT_USE_NEON="yes"
              PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              ;;
            riscv64)
              PHP_ORT_USE_RISCV64="yes"
              PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              ;;
            avx512)
              PHP_ORT_USE_AVX512="yes"
              PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              ;;
            avx2)
              PHP_ORT_USE_AVX2="yes"
              PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              ;;
            sse41)
              PHP_ORT_USE_SSE41="yes"
              PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              ;;
            sse2)
              PHP_ORT_USE_SSE2="yes"
              PHP_ORT_CPU_BACKEND_COUNT=`expr $PHP_ORT_CPU_BACKEND_COUNT + 1`
              ;;
            none)
              ;;
            *)
              IFS="$saved_IFS"
              AC_MSG_ERROR([Invalid backend: $backend. Valid: cuda,auto,wasm,neon,riscv64,avx512,avx2,sse41,sse2,none])
              ;;
          esac
        done
        IFS="$saved_IFS"
        
        dnl Enforce: only one CPU backend allowed
        if test $PHP_ORT_CPU_BACKEND_COUNT -gt 1; then
          AC_MSG_ERROR([Only one CPU backend allowed (found $PHP_ORT_CPU_BACKEND_COUNT). Example: cuda,avx2 not cuda,avx2,sse41])
        fi
        ;;
    esac

    dnl Configure selected backends
    if test "$PHP_ORT_BACKEND" != "no"; then
      dnl CUDA backend
      if test "$PHP_ORT_USE_CUDA" = "yes"; then
        if test "$PHP_ORT_HAS_CUDA" != "yes"; then
          AC_MSG_ERROR([CUDA backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -I$CUDA_DIR/include"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL CUDA"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
          $PHP_ORT_BACKEND_DIR/cuda/abs.c
          $PHP_ORT_BACKEND_DIR/cuda/add.c
          $PHP_ORT_BACKEND_DIR/cuda/ceil.c
          $PHP_ORT_BACKEND_DIR/cuda/div.c
          $PHP_ORT_BACKEND_DIR/cuda/dot.c
          $PHP_ORT_BACKEND_DIR/cuda/floor.c
          $PHP_ORT_BACKEND_DIR/cuda/matmul.c
          $PHP_ORT_BACKEND_DIR/cuda/mul.c
          $PHP_ORT_BACKEND_DIR/cuda/neg.c
          $PHP_ORT_BACKEND_DIR/cuda/recip.c
          $PHP_ORT_BACKEND_DIR/cuda/round.c
          $PHP_ORT_BACKEND_DIR/cuda/sign.c
          $PHP_ORT_BACKEND_DIR/cuda/sqrt.c
          $PHP_ORT_BACKEND_DIR/cuda/sub.c
          $PHP_ORT_BACKEND_DIR/cuda/trunc.c
          $PHP_ORT_BACKEND_DIR/cuda/impl.c"

        PHP_ADD_LIBRARY_WITH_PATH(cudart, $CUDA_DIR/lib64, ORT_SHARED_LIBADD)
        PHP_ADD_LIBRARY_WITH_PATH(cudart, $CUDA_DIR/lib, ORT_SHARED_LIBADD)
        PHP_ADD_LIBRARY_WITH_PATH(cublas, $CUDA_DIR/lib64, ORT_SHARED_LIBADD)
        PHP_ADD_LIBRARY_WITH_PATH(cublas, $CUDA_DIR/lib, ORT_SHARED_LIBADD)
        PHP_ADD_LIBRARY_WITH_PATH(kernels, $PHP_ORT_BACKEND_DIR/cuda, ORT_SHARED_LIBADD)

        AC_DEFINE(ORT_BACKEND_GPU_ENABLED, 1, [GPU backend enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_GPU_NAME, "CUDA", [GPU Backend Name])
      fi

      dnl WASM backend
      if test "$PHP_ORT_USE_WASM" = "yes"; then
        if test "$PHP_ORT_HAS_WASM" != "yes"; then
          AC_MSG_ERROR([WASM backend requested but not available])
        fi
        if test "$abs_srcdir" != "$abs_builddir"; then
          AC_MSG_ERROR([WASM backend only supports in-tree builds])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -msimd128"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL WASM"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
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
          $PHP_ORT_BACKEND_DIR/wasm/impl.c"
        AC_DEFINE(ORT_BACKEND_CPU_ENABLED, 1, [Backend CPU optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_CPU_NAME, "WASM", [CPU Backend Name])
        PHP_ORT_POOL="no"
        ext_shared=no
      fi

      dnl NEON backend
      if test "$PHP_ORT_USE_NEON" = "yes"; then
        if test "$PHP_ORT_HAS_NEON" != "yes"; then
          AC_MSG_ERROR([NEON backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -march=armv8-a+simd"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL NEON"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
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
          $PHP_ORT_BACKEND_DIR/neon/impl.c"
        AC_DEFINE(ORT_BACKEND_CPU_ENABLED, 1, [Backend CPU optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_CPU_NAME, "NEON", [CPU Backend Name])
      fi

      dnl RISC-V 64 backend
      if test "$PHP_ORT_USE_RISCV64" = "yes"; then
        if test "$PHP_ORT_HAS_RISCV64" != "yes"; then
          AC_MSG_ERROR([RISC-V 64 backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -march=rv64gcv"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL RISCV64"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
          $PHP_ORT_BACKEND_DIR/riscv64/abs.c
          $PHP_ORT_BACKEND_DIR/riscv64/add.c
          $PHP_ORT_BACKEND_DIR/riscv64/div.c
          $PHP_ORT_BACKEND_DIR/riscv64/matmul.c
          $PHP_ORT_BACKEND_DIR/riscv64/mul.c
          $PHP_ORT_BACKEND_DIR/riscv64/neg.c
          $PHP_ORT_BACKEND_DIR/riscv64/recip.c
          $PHP_ORT_BACKEND_DIR/riscv64/sign.c
          $PHP_ORT_BACKEND_DIR/riscv64/sqrt.c
          $PHP_ORT_BACKEND_DIR/riscv64/sub.c
          $PHP_ORT_BACKEND_DIR/riscv64/impl.c"
        AC_DEFINE(ORT_BACKEND_CPU_ENABLED, 1, [Backend CPU optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_CPU_NAME, "RISCV64", [CPU Backend Name])
      fi

      dnl AVX512 backend
      if test "$PHP_ORT_USE_AVX512" = "yes"; then
        if test "$PHP_ORT_HAS_AVX512" != "yes"; then
          AC_MSG_ERROR([AVX512 backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -mavx512f -mavx512bw -mavx512dq -mavx512vl -mevex512"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL AVX512"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
          $PHP_ORT_BACKEND_DIR/avx512/add.c
          $PHP_ORT_BACKEND_DIR/avx512/sub.c
          $PHP_ORT_BACKEND_DIR/avx512/dot.c
          $PHP_ORT_BACKEND_DIR/avx512/matmul.c
          $PHP_ORT_BACKEND_DIR/avx512/mul.c
          $PHP_ORT_BACKEND_DIR/avx512/div.c
          $PHP_ORT_BACKEND_DIR/avx512/sqrt.c
          $PHP_ORT_BACKEND_DIR/avx512/neg.c
          $PHP_ORT_BACKEND_DIR/avx512/ceil.c
          $PHP_ORT_BACKEND_DIR/avx512/floor.c
          $PHP_ORT_BACKEND_DIR/avx512/round.c
          $PHP_ORT_BACKEND_DIR/avx512/abs.c
          $PHP_ORT_BACKEND_DIR/avx512/sign.c
          $PHP_ORT_BACKEND_DIR/avx512/recip.c
          $PHP_ORT_BACKEND_DIR/avx512/trunc.c
          $PHP_ORT_BACKEND_DIR/avx512/impl.c"
        AC_DEFINE(ORT_BACKEND_CPU_ENABLED, 1, [Backend CPU optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_CPU_NAME, "AVX512", [CPU Backend Name])
      fi

      dnl AVX2 backend
      if test "$PHP_ORT_USE_AVX2" = "yes"; then
        if test "$PHP_ORT_HAS_AVX2" != "yes"; then
          AC_MSG_ERROR([AVX2 backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -mavx2"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL AVX2"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
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
          $PHP_ORT_BACKEND_DIR/avx2/impl.c"
        AC_DEFINE(ORT_BACKEND_CPU_ENABLED, 1, [Backend CPU optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_CPU_NAME, "AVX2", [CPU Backend Name])
      fi

      dnl SSE4.1 backend
      if test "$PHP_ORT_USE_SSE41" = "yes"; then
        if test "$PHP_ORT_HAS_SSE41" != "yes"; then
          AC_MSG_ERROR([SSE4.1 backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -msse4.1"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL SSE4.1"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
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
          $PHP_ORT_BACKEND_DIR/sse41/impl.c"
        AC_DEFINE(ORT_BACKEND_CPU_ENABLED, 1, [Backend CPU optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_CPU_NAME, "SSE4.1", [CPU Backend Name])
      fi

      dnl SSE2 backend
      if test "$PHP_ORT_USE_SSE2" = "yes"; then
        if test "$PHP_ORT_HAS_SSE2" != "yes"; then
          AC_MSG_ERROR([SSE2 backend requested but not available])
        fi
        PHP_ORT_BACKEND_CFLAGS="$PHP_ORT_BACKEND_CFLAGS -msse2"
        PHP_ORT_BACKEND_LEVEL="$PHP_ORT_BACKEND_LEVEL SSE2"
        PHP_ORT_BACKEND_IMPL="$PHP_ORT_BACKEND_IMPL
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
          $PHP_ORT_BACKEND_DIR/sse2/impl.c"
        AC_DEFINE(ORT_BACKEND_CPU_ENABLED, 1, [Backend CPU optimizations enabled])
        AC_DEFINE_UNQUOTED(ORT_BACKEND_CPU_NAME, "SSE2", [CPU Backend Name])
      fi

      dnl Normalize backend implementation list
      PHP_ORT_BACKEND_IMPL=m4_normalize("$PHP_ORT_BACKEND_IMPL")

      dnl Report configured backends
      AC_MSG_CHECKING([for enabled backends])
      if test -n "$PHP_ORT_BACKEND_LEVEL"; then
        AC_MSG_RESULT([$PHP_ORT_BACKEND_LEVEL])
        AC_MSG_CHECKING([for backend CFLAGS])
        AC_MSG_RESULT([$PHP_ORT_BACKEND_CFLAGS])
        PHP_SUBST(PHP_ORT_BACKEND_CFLAGS)
      else
        AC_MSG_RESULT([none])
        PHP_ORT_BACKEND="no"
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
  dnl Coverage Support
  dnl ============================================================
  AC_MSG_CHECKING([for coverage support])
  AS_VAR_IF([PHP_ORT_COVERAGE], [no], [
    AC_MSG_RESULT([no])
  ], [
    AC_DEFINE(HAVE_ORT_COVERAGE, 1,
      [Defined to 1 where we are generating coverage])
    EXTRA_CFLAGS="$EXTRA_CFLAGS -fprofile-arcs -ftest-coverage"
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -fprofile-arcs"

    dnl On non-Darwin systems, we need to explicitly link libgcov
    case $host_os in
      darwin*)
        ;;
      *)
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lgcov"
        ;;
    esac

    PHP_SUBST(EXTRA_CFLAGS)
    PHP_SUBST(EXTRA_LDFLAGS)
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

  if test -n "$PHP_ORT_BACKEND_LEVEL"; then
    PHP_ADD_INCLUDE([$ext_builddir/$PHP_ORT_BACKEND_DIR])
    PHP_ADD_BUILD_DIR([$ext_builddir/$PHP_ORT_BACKEND_DIR])
  fi
  
  PHP_SUBST([ORT_SHARED_LIBADD])
  PHP_ADD_MAKEFILE_FRAGMENT(Makefile.frag)
])