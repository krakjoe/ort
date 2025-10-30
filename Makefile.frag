ort-test-coverage:
	CCACHE_DISABLE=1 EXTRA_CFLAGS="-fprofile-arcs -ftest-coverage" TEST_PHP_ARGS="-q" $(MAKE) clean test

ort-test-coverage-lcov: ort-test-coverage
	lcov --directory $(top_srcdir)/src             \
		 --no-external                             \
		 --output-file $(top_srcdir)/coverage.info \
		 -c

ort-test-coverage-html: ort-test-coverage-lcov
	genhtml $(top_srcdir)/coverage.info --output-directory=$(top_srcdir)/html

################################################################################
# CUDA
################################################################################
ORT_CUDA_SRCDIR = $(top_srcdir)/src/maths/backend/cuda/kernels
ORT_CUDA_OBJDIR = $(top_builddir)/src/maths/backend/cuda/kernels
ORT_CUDA_LIBDIR = $(top_builddir)/src/maths/backend/cuda

ORT_CUDA_SOURCES = $(wildcard $(ORT_CUDA_SRCDIR)/*.cu)
ORT_CUDA_OBJECTS = $(patsubst $(ORT_CUDA_SRCDIR)/%.cu,$(ORT_CUDA_OBJDIR)/%.o,$(ORT_CUDA_SOURCES))
ORT_CUDA_LIBRARY = $(ORT_CUDA_LIBDIR)/libkernels.a

ORT_CUDA_CFLAGS = -std=c++14 \
				  -O3 \
				  --compiler-options=-fPIC \
				  -arch=sm_60 \
				  -I$(top_srcdir)/src

ort-cuda-kernels: $(ORT_CUDA_LIBRARY)

$(ORT_CUDA_LIBRARY): $(ORT_CUDA_OBJECTS)
	@echo "Creating CUDA kernel library"
	$(AR) rcs $@ $^
	@echo "CUDA kernels built: $@"

$(ORT_CUDA_OBJDIR)/%.o: $(ORT_CUDA_OBJDIR)/%.cu
	$(NVCC) $(ORT_CUDA_CFLAGS) -c $< -o $@

.PHONY: ort-cuda-kernels ort-test-coverage ort-test-coverage-lcov ort-test-coverage-html