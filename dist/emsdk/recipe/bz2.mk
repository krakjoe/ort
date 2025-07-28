########################################################################
# ort                                                                  #
########################################################################
# Copyright (c) Joe Watkins 2025                                       #
########################################################################
# This source file is subject to version 3.01 of the PHP license,      #
# that is bundled with this package in the file LICENSE, and is        #
# available through the world-wide-web at the following url:           #
# http://www.php.net/license/3_01.txt                                  #
# If you did not receive a copy of the PHP license and are unable to   #
# obtain it through the world-wide-web, please send a note to          #
# license@php.net so we can mail you a copy immediately.               #
########################################################################
# Author: krakjoe                                                      #
########################################################################
EM_BZ2_URL   = https://www.sourceware.org/pub/bzip2/bzip2-latest.tar.gz
EM_BZ2_TAR   = $(EM_EXTRA_RECIPE_OUT)/bz2-1.3.1.tar.gz
EM_BZ2_SRC   = $(EM_EXTRA_RECIPE_OUT)/bz2
EM_BZ2_LIB   = $(EM_BZ2_SRC)/lib/libbz2.a
EM_BZ2_INC   = $(EM_BZ2_SRC)/include
########################################################################
EM_BZ2_CFLAGS ?= -Wall -Winline -O2
########################################################################
.PHONY: all-bz2 clean-bz2
########################################################################
all-bz2: $(EM_BZ2_LIB)

$(EM_BZ2_TAR):
	wget $(EM_BZ2_URL) -O $(EM_BZ2_TAR)
	mkdir -p $(EM_BZ2_SRC)
	tar -C $(EM_BZ2_SRC) --strip-components=1 -xvzf \
		$(EM_BZ2_TAR)

$(EM_BZ2_LIB): $(EM_BZ2_TAR)
	mkdir -p $(EM_BZ2_SRC)/lib
	$(CC) $(EM_BZ2_CFLAGS) -c $(EM_BZ2_SRC)/blocksort.c     -o $(EM_BZ2_SRC)/lib/blocksort.o
	$(CC) $(EM_BZ2_CFLAGS) -c $(EM_BZ2_SRC)/huffman.c       -o $(EM_BZ2_SRC)/lib/huffman.o
	$(CC) $(EM_BZ2_CFLAGS) -c $(EM_BZ2_SRC)/crctable.c      -o $(EM_BZ2_SRC)/lib/crctable.o
	$(CC) $(EM_BZ2_CFLAGS) -c $(EM_BZ2_SRC)/randtable.c     -o $(EM_BZ2_SRC)/lib/randtable.o
	$(CC) $(EM_BZ2_CFLAGS) -c $(EM_BZ2_SRC)/compress.c      -o $(EM_BZ2_SRC)/lib/compress.o
	$(CC) $(EM_BZ2_CFLAGS) -c $(EM_BZ2_SRC)/decompress.c    -o $(EM_BZ2_SRC)/lib/decompress.o
	$(CC) $(EM_BZ2_CFLAGS) -c $(EM_BZ2_SRC)/bzlib.c         -o $(EM_BZ2_SRC)/lib/bzlib.o
	$(AR) cq $(EM_BZ2_LIB)                   \
		$(EM_BZ2_SRC)/lib/blocksort.o        \
		$(EM_BZ2_SRC)/lib/huffman.o          \
		$(EM_BZ2_SRC)/lib/crctable.o         \
		$(EM_BZ2_SRC)/lib/randtable.o        \
		$(EM_BZ2_SRC)/lib/compress.o         \
		$(EM_BZ2_SRC)/lib/decompress.o       \
		$(EM_BZ2_SRC)/lib/bzlib.o
	mkdir -p $(EM_BZ2_SRC)/include
	install $(EM_BZ2_SRC)/bzlib.h $(EM_BZ2_SRC)/include
	$(RANLIB) $(EM_BZ2_LIB)

clean-bz2:
	@rm -rf $(EM_BZ2_TAR)
	@rm -rf $(EM_BZ2_SRC)

########################################################################
# Inject for em
########################################################################
EM_EXTRA_RECIPE_TARGETS  += $(EM_BZ2_LIB)
EM_EXTRA_RECIPE_CLEANERS += clean-bz2
EM_EXTRA_CONFIGURE       += --with-bz2=$(EM_BZ2_SRC)
EM_EXTRA_LINK            += $(EM_BZ2_LIB)
########################################################################
# Inject for autoconf
########################################################################
export ac_cv_lib_bz2_BZ2_bzerror=yes