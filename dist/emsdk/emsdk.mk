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
# Private, toolchain, unlikely to be necessary to set any of this
########################################################################
export CC=emcc
export CXX=em++
export AR=emar
export RANLIB=emranlib
export STRIP=emstrip
export CFLAGS=-DEMSCRIPTEN -DHAVE_REALLOCARRAY -O2
export EXTRA_LIBS=emsdk.stub.lo
########################################################################
# Super duper private, probably stuff will break if caller sets these
########################################################################
export EMCONFIGURE ?= emconfigure
export EMMAKE      ?= emmake
export EMCFLAGS    ?= -Imain -IZend -ITSRM -Iext/standard -I.
########################################################################
# Protected, should probably not be set by caller, may be set by recipes
########################################################################
EM_EXTRA_TMP       ?= /tmp
EM_EXTRA_NPROC     ?= $(nproc)
EM_EXTRA_CONFIGURE ?=
EM_EXTRA_COMPILE   ?=
EM_EXTRA_LINK      ?=
########################################################################
# Public, may be set by caller
########################################################################
EM_EXTRA_RECIPE_IN      ?= recipe
EM_EXTRA_RECIPE_OUT     ?= /tmp
########################################################################
# Private, used in recipes
########################################################################
EM_EXTRA_RECIPE_TARGETS  ?=
EM_EXTRA_RECIPE_CLEANERS ?=
########################################################################
LIBTOOL     ?= $(realpath libtool)
########################################################################
# Load recipes
########################################################################
ifneq ($(filter with-%, $(MAKECMDGOALS)),)
EM_RECIPES      := $(patsubst with-%,%,$(filter with-%, $(MAKECMDGOALS)))
EM_RECIPE_PATHS := $(foreach r,$(EM_RECIPES),$(EM_EXTRA_RECIPE_IN)/$(r).mk)
$(foreach p,$(EM_RECIPE_PATHS), \
  $(if $(wildcard $(p)), \
    ,$(error Recipe: $(p) not found in recipe/) \
  ) \
)
include $(EM_RECIPE_PATHS)
endif
########################################################################
.PHONY: all clean clean-emsdk clean-wasm clean-recipes strip install with-%
########################################################################
with-%:
	@true
########################################################################
all: wasm

config.status: $(EM_EXTRA_RECIPE_TARGETS)
	@./buildconf --force
	@$(EMCONFIGURE) ./configure \
		--disable-all \
		--disable-cgi \
		--disable-cli \
		--disable-phpdbg \
		--enable-embed=static \
		--disable-fiber-asm \
		--without-pcre-jit \
		$(EM_EXTRA_CONFIGURE)

emsdk.stub.lo: emsdk.stub.c config.status
	$(LIBTOOL) --silent --mode=compile --tag=CC \
		$(CC) $(EMCFLAGS) $(EM_EXTRA_COMPILE) \
			-c emsdk.stub.c -o emsdk.stub.lo

.libs/libphp.a.stamp: emsdk.stub.lo
	@$(EMMAKE) make -j$(EM_EXTRA_NPROC)
	@touch $@

.libs/libphp.a: .libs/libphp.a.stamp
	@true

emsdk.io.lo: emsdk.io.c .libs/libphp.a
	$(LIBTOOL) --silent --mode=compile --tag=CC \
		$(CC) $(EMCFLAGS) $(EM_EXTRA_COMPILE) \
			-c emsdk.io.c -o emsdk.io.lo

emsdk.api.lo: emsdk.api.c emsdk.io.lo .libs/libphp.a
	$(LIBTOOL) --silent --mode=compile --tag=CC \
		$(CC) $(EMCFLAGS) $(EM_EXTRA_COMPILE) \
			-c emsdk.api.c -o emsdk.api.lo

build: .libs/libphp.a

api: emsdk.api.lo

wasm: emsdk.api.lo .libs/libphp.a
	$(LIBTOOL) --silent --preserve-dup-deps --mode=link --tag=CC \
	$(CC) -o php-em.js --post-js=emsdk.stub.js \
		-s EXPORTED_FUNCTIONS='["_em_startup", "_em_run_string", "_em_run_length", "_em_run_free", "_em_shutdown", "_em_io_buffer"]' \
		-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8", "lengthBytesUTF8", "HEAPU8"]' \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s INITIAL_MEMORY=128MB \
		-s WASM=1 \
		$(EM_EXTRA_LINK) \
		.libs/libphp.a \
		emsdk.stub.o \
		emsdk.io.o \
		emsdk.api.o
	@ls -lash php-em.js php-em.wasm

strip: wasm
	$(STRIP) php-em.wasm

install: wasm
	@$(EMMAKE) make install

clean-recipes: $(EM_EXTRA_RECIPE_CLEANERS)

clean-emsdk:
	@rm -rf emsdk.*.o
	@rm -rf emsdk.*.lo

clean-wasm:
	@rm -rf php-em.js
	@rm -rf php-em.wasm

clean-php:
	@rm -rf config.status
	@rm -f .libs/libphp.a.stamp

clean: clean-recipes clean-emsdk clean-wasm clean-php
	@$(EMMAKE) make clean