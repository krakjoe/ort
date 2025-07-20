/*
  +----------------------------------------------------------------------+
  | ort                                                                  |
  +----------------------------------------------------------------------+
  | Copyright (c) Joe Watkins 2025                                       |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: krakjoe                                                      |
  +----------------------------------------------------------------------+
 */

/* {{{ 
    @file generators/random.h
    @brief This file contains symbols for internal random generator registration
}}}*/

#ifndef HAVE_ORT_GENERATORS_RANDOM
#define HAVE_ORT_GENERATORS_RANDOM

#include "ort.h"

extern zend_class_entry* php_ort_generator_random_int_ce;

PHP_MINIT_FUNCTION(ORT_GENERATORS_RANDOM);
PHP_MSHUTDOWN_FUNCTION(ORT_GENERATORS_RANDOM);
#endif