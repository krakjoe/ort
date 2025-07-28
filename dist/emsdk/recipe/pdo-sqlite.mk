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
# Nothing to do ...
########################################################################
# Inject for em
########################################################################
EM_EXTRA_CONFIGURE       += --with-pdo-sqlite=static
########################################################################
# Inject for autoconf
########################################################################
export ac_cv_path_SQLITE_PKG_CONFIG=/bin/false
export ac_cv_lib_sqlite3_sqlite3_close_v2=yes
export ac_cv_lib_sqlite3_sqlite3_column_table_name=no
