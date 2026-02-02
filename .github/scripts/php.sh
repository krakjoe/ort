#!/bin/bash
###############################################################################
# Build PHP for CI
#
# By default will build master in /opt/php/$PHP_MASTER from a single branch
# clone
###############################################################################
PHP_MASTER=8.6
###############################################################################
PHP_VERSION=${PHP_VERSION:-$PHP_MASTER}
###############################################################################
PHP_SRC=${PHP_SRC:-/opt/src/php-build/$PHP_VERSION}
###############################################################################
PHP_ARCH=${PHP_ARCH:-$(uname -m)}
###############################################################################
PHP_PREFIX=${PHP_PREFIX:-/opt/php/$PHP_VERSION}
###############################################################################
PHP_CONFIG_PATH=${PHP_CONFIG_PATH:-$PHP_PREFIX/etc}
###############################################################################
PHP_CONFIG_SCAN=${PHP_CONFIG_SCAN:-$PHP_CONFIG_PATH/php.d}
###############################################################################
PHP_CONFIG_CACHE=${PHP_CONFIG_CACHE:-~/php.cache/$PHP_ARCH-$PHP_VERSION.config}
###############################################################################
echo "[php.build] CPU Configuration for $PHP_VERSION-$PHP_ARCH"

which lscpu >/dev/null

if [ $? -eq 0 ];
then
	lscpu
else
	which sysctl >/dev/null

	if [ $? -eq 0 ];
	then
		sysctl -n machdep.cpu.brand_string
	else
		echo "[php.build] could not determine CPU Configuration"
	fi
fi
###############################################################################
rm -rf $PHP_SRC

if [ $PHP_VERSION == $PHP_MASTER ];
then
	PHP_BRANCH=master
else
	PHP_BRANCH=PHP-$PHP_VERSION
fi

git clone -b $PHP_BRANCH --single-branch https://github.com/php/php-src $PHP_SRC
###############################################################################
cd $PHP_SRC

if [ $? -ne 0 ];
then
	echo "[php.build] 404 PHP-$PHP_VERSION for $PHP_ARCH in $PHP_SRC"
	exit 1
fi

if [ ! -f $PHP_CONFIG_CACHE ];
then
	mkdir -p $(dirname $PHP_CONFIG_CACHE)
fi
###############################################################################
echo "[php.build] Rebuild $PHP_VERSION for $PHP_ARCH in $PHP_SRC"

$PHP_SRC/buildconf --force
$PHP_SRC/configure \
	--prefix=$PHP_PREFIX \
	--disable-all \
	--disable-phpdbg \
	--disable-cgi \
	--enable-gd \
	--with-jpeg \
	--enable-tokenizer \
	--enable-ctype \
	--enable-filter \
	--with-config-file-path=$PHP_CONFIG_PATH \
	--with-config-file-scan-dir=$PHP_CONFIG_SCAN \
	--enable-zts \
	--with-curl=static \
	--enable-phar \
	--with-zlib \
	--cache-file=$PHP_CONFIG_CACHE \
	--without-pcre-jit \
	$PHP_CONFIG_OPTS

if [ $? -ne 0 ];
then
	echo "[php.build] Configuration Failed $PHP_VERSION for $PHP_ARCH in $PHP_SRC"
	exit 1
fi
###############################################################################
make -j$(nproc)

if [ $? -ne 0 ];
then
	echo "[php.build] Build Failed $PHP_VERSION for $PHP_ARCH in $PHP_SRC"
	exit 1
fi
###############################################################################
make install

if [ $? -ne 0 ];
then
	echo "[php.build] Install Failed $PHP_VERSION for $PHP_ARCH in $PHP_SRC"
	exit 1
fi

echo "[php.build] Complete $PHP_VERSION for $PHP_ARCH"
###############################################################################