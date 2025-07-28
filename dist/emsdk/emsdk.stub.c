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

#include <sys/types.h>
#include <stdint.h>
#include <unistd.h>
#include <spawn.h>
#include <ucontext.h>
#include <stdlib.h>

int getdtablesize(void) {
    return 1024; // reasonable default
}

int posix_spawnp(pid_t *pid, const char *file,
                const posix_spawn_file_actions_t *actions,
                const posix_spawnattr_t *attributes,
                char *const argv[], char *const env[]) {
    (void) pid;
    (void) file;
    (void) actions;
    (void) attributes;
    (void) argv;
    (void) env;

    return -1; // not supported
}

void *reallocarray(void *ptr, size_t nmemb, size_t size) {
    if (nmemb && size && SIZE_MAX / nmemb < size) {
        return NULL;
    }
    return realloc(ptr, nmemb * size);
}

int getcontext(ucontext_t *ucp) {
    (void) ucp;

    return -1; // not supported, but fibers should handle this gracefully
}

void makecontext(ucontext_t *ucp, void (*func)(void), int argc, ...) {
    (void) ucp;
    (void) func;
    (void) argc;

    // stub - fibers will fall back to alternative implementation
}

int swapcontext(ucontext_t *oucp, const ucontext_t *ucp) {
    (void) oucp;
    (void) ucp;

    return -1; // not supported
}

int munmap(void *addr, size_t length) {
    (void)addr;    
    (void)length;  

    return 0; // not supported
}