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

#include <php.h>

#ifdef ZTS
#include <TSRM.h>
#endif

#ifdef EMSCRIPTEN
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <SAPI.h>

#include <php_main.h>
#include <zend_exceptions.h>

typedef struct _em_io_abstract_t {
    char*   data;
    ssize_t length;
    size_t  position;
} em_io_abstract_t;

void* EMSCRIPTEN_KEEPALIVE em_io_buffer(uintptr_t ab, size_t length) {
    em_io_abstract_t* abstract =
        (em_io_abstract_t*) ab;

    abstract->data     = pecalloc(sizeof(char), length, 1);
    abstract->length   = length;
    abstract->position = 0;

    return abstract->data;
}

EM_JS(ssize_t, em_io_fetch, (const char* url, uintptr_t abstract), {
    var url_str = UTF8ToString(url);
    var xhr = new XMLHttpRequest();

    if (Module.dispatchEvent) {
        Module.dispatchEvent(new CustomEvent('io.begin', { 
            "detail": { 
                "url": url_str,
                "xhr": xhr}
        }));
    }

    try {
        xhr.open('GET', url_str, false);
        xhr.send();

        if (xhr.status >= 200 && xhr.status < 300) {
            var response = xhr.responseText;
            var length = lengthBytesUTF8(response) + 1;
            var buffer = Module._em_io_buffer(abstract, length);

            stringToUTF8(
                response, buffer,length);

            if (Module.dispatchEvent) {
                Module.dispatchEvent(new CustomEvent('io.end', { 
                    "detail": { 
                        "url":    url_str,
                        "xhr":    xhr,
                        "buffer": buffer,
                        "length": length}
                }));
            }

            return length;
        } else {
            if (Module.dispatchEvent) {
                Module.dispatchEvent(new CustomEvent('io.error', { 
                    "detail": { 
                        "url": url_str,
                        "xhr": xhr}
                }));
            }
        }
    } catch (exception) {
        if (Module.dispatchEvent) {
            Module.dispatchEvent(new CustomEvent('io.exception', {
                "detail": { 
                    "url":       url_str,
                    "xhr":       xhr,
                    "exception": exception,
                }
            }));
        }

        console.error('Fetch failed:', exception);
    }
    return -1;
});

// Abstract creation of stream
em_io_abstract_t* em_io_abstract(const char* path) {
    em_io_abstract_t* abstract = 
        (em_io_abstract_t*)
            pecalloc(1, sizeof(em_io_abstract_t), 1);

    abstract->length = em_io_fetch(
        path, (uintptr_t) abstract);
    abstract->position = 0;

    return abstract;
}

// Stream wrapper operations
static ssize_t em_io_read(php_stream *stream, char *buffer, size_t count) {
    em_io_abstract_t* abstract =
        (em_io_abstract_t*)
            stream->abstract;

    if (abstract->length <= 0) {
        return abstract->length;
    }

    if (count > abstract->length - abstract->position) {
        count = abstract->length - abstract->position;
    }

    memcpy(buffer,
        &abstract->data[abstract->position],
        count);

    abstract->position += count;

    return count;
}

static int em_io_close(php_stream *stream, int type) {
    em_io_abstract_t* abstract =
        (em_io_abstract_t*) stream->abstract;

    if (abstract) {
        if (abstract->data) {
            pefree(abstract->data, 1);
        }
        pefree(abstract, 1);
    }
    return 0;
}

static php_stream_ops em_io_ops = {
    NULL,          // write
    em_io_read,    // read  
    em_io_close,   // close
    NULL,          // flush
    "em-io",
    NULL,          // seek
    NULL,          // cast
    NULL,          // stat
    NULL           // set_option
};

static php_stream *em_io_opener(php_stream_wrapper *wrapper, 
                                  const char *path, const char *mode,
                                  int options, zend_string **opened_path,
                                  php_stream_context *context) {
    em_io_abstract_t* abstract = em_io_abstract(path);

    if (!abstract || abstract->length < 0) {
        if (abstract) {
            if (abstract->data) {
                pefree(abstract->data, 1);
            }
            pefree(abstract, 1); 
        }
        return NULL; 
    }
    return php_stream_alloc(&em_io_ops, abstract, 0, mode);
}

static php_stream_wrapper_ops em_io_wrapper_ops = {
    em_io_opener,
    NULL, // close
    NULL, // fstat
    NULL, // stat
    NULL, // opendir
    "em-io"
};

static php_stream_wrapper em_io_wrapper = {
    &em_io_wrapper_ops,
    NULL,
    0
};

void em_io_activate(void) {
    php_unregister_url_stream_wrapper("http");
    php_unregister_url_stream_wrapper("https");

    php_register_url_stream_wrapper("http", &em_io_wrapper);
    php_register_url_stream_wrapper("https", &em_io_wrapper);
}