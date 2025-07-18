ort-test-coverage:
	CCACHE_DISABLE=1 EXTRA_CFLAGS="-fprofile-arcs -ftest-coverage" TEST_PHP_ARGS="-q" $(MAKE) clean test

ort-test-coverage-lcov: ort-test-coverage
	lcov --directory $(top_srcdir)/src             \
		 --no-external                             \
		 --output-file $(top_srcdir)/coverage.info \
		 -c

ort-test-coverage-html: ort-test-coverage-lcov
	genhtml $(top_srcdir)/coverage.info --output-directory=$(top_srcdir)/html
