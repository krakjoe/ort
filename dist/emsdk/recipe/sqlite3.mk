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
EM_SQLITE3_URL   = https://sqlite.org/src/tarball/sqlite.tar.gz?r=release
EM_SQLITE3_TAR   = $(EM_EXTRA_RECIPE_OUT)/sqlite3.tar.gz
EM_SQLITE3_SRC   = $(EM_EXTRA_RECIPE_OUT)/sqlite3
EM_SQLITE3_LIB   = $(EM_SQLITE3_SRC)/lib/libsqlite3.a
EM_SQLITE3_INC   = $(EM_SQLITE3_SRC)/include
########################################################################
EM_SQLITE3_CFLAGS    ?=
EM_SQLITE3_LDFLAGS   ?=
EM_SQLITE3_CONFIGURE ?=
########################################################################
EM_SQLITE3_CONFIGURED  = $(EM_SQLITE3_SRC)/Makefile
EM_SQLITE3_MADE        = $(EM_SQLITE3_SRC)/libsqlite3.a
########################################################################
.PHONY: all-sqlite3 clean-sqlite3
########################################################################
all-sqlite3: install-sqlite3

$(EM_SQLITE3_TAR):
	wget $(EM_SQLITE3_URL) -O $(EM_SQLITE3_TAR)
	mkdir -p $(EM_SQLITE3_SRC)
	tar -C $(EM_SQLITE3_SRC) --strip-components=1 -xvzf \
		$(EM_SQLITE3_TAR)

$(EM_SQLITE3_CONFIGURED): $(EM_SQLITE3_TAR)
	cd $(EM_SQLITE3_SRC) && \
	CFLAGS=$(EM_SQLITE3_CFLAGS) LDFLAGS=$(EM_SQLITE3_LDFLAGS) \
		$(EMCONFIGURE) ./configure \
			--disable-threadsafe \
			--disable-readline \
			--disable-load-extension \
			--disable-tcl \
			--disable-shared \
			--enable-static \
			--prefix=$(EM_SQLITE3_SRC) \
			$(EM_SQLITE3_CONFIGURE)
	@touch $@

$(EM_SQLITE3_MADE): $(EM_SQLITE3_CONFIGURED)
	cd $(EM_SQLITE3_SRC) && \
	CFLAGS=$(EM_SQLITE3_CFLAGS) LDFLAGS=$(EM_SQLITE3_LDFLAGS) \
		$(EMMAKE) make -j$(EM_EXTRA_NPROC)

$(EM_SQLITE3_LIB): $(EM_SQLITE3_MADE)
	cd $(EM_SQLITE3_SRC) && \
	CFLAGS=$(EM_SQLITE3_CFLAGS) LDFLAGS=$(EM_SQLITE3_LDFLAGS) \
		$(EMMAKE) make install

clean-sqlite3:
	@rm -rf $(EM_SQLITE3_TAR)
	@rm -rf $(EM_SQLITE3_SRC)

########################################################################
# Inject for em
########################################################################
EM_EXTRA_RECIPE_TARGETS  += $(EM_SQLITE3_LIB)
EM_EXTRA_RECIPE_CLEANERS += clean-sqlite3
EM_EXTRA_CONFIGURE       += --with-sqlite3=static,$(EM_SQLITE3_SRC)
EM_EXTRA_LINK            += $(EM_SQLITE3_LIB)
########################################################################
# Export for php
########################################################################
export SQLITE_LIBS       = $(EM_SQLITE3_LIB)
export SQLITE_CFLAGS     = -I$(EM_SQLITE3_INC)
########################################################################
# Inject for autoconf
########################################################################
export ac_cv_lib_sqlite3_sqlite3_errstr=yes
export ac_cv_lib_sqlite3_sqlite3_expanded_sql=yes
export ac_cv_lib_sqlite3_sqlite3_load_extension=no