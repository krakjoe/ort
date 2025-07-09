
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

#include "maths/cast.h"
#include "maths/pool.h"
#include "maths/result.h"

#if defined(_WIN32)
# include <windows.h>
# define ORT_POOL_LOCAL __declspec(thread)
  typedef HANDLE ort_thread_t;
  typedef CRITICAL_SECTION ort_mutex_t;
  typedef CONDITION_VARIABLE ort_cond_t;
#define ort_pool_mutex_init(mutex) \
    InitializeCriticalSection(mutex)
#define ort_pool_mutex_lock(mutex) \
    EnterCriticalSection(mutex)
#define ort_pool_mutex_unlock(mutex) \
    LeaveCriticalSection(mutex)
#define ort_pool_mutex_destroy(mutex) \
    DeleteCriticalSection(mutex)
#define ort_pool_cond_init(cond) \
    InitializeConditionVariable(cond)
#define ort_pool_cond_wait(cond, mutex) \
    SleepConditionVariableCS(cond, mutex, INFINITE)
#define ort_pool_cond_signal(cond) \
    WakeConditionVariable(cond)
#define ort_pool_cond_broadcast(cond) \
    WakeAllConditionVariable(cond)
#define ort_pool_cond_destroy(cond)
#else
# include <pthread.h>
# include <unistd.h>
# define ORT_POOL_LOCAL __thread
  typedef pthread_t ort_thread_t;
  typedef pthread_mutex_t ort_mutex_t;
  typedef pthread_cond_t ort_cond_t;
#define ort_pool_mutex_init(mutex) \
    pthread_mutex_init(mutex, NULL)
#define ort_pool_mutex_lock(mutex) \
    pthread_mutex_lock(mutex)
#define ort_pool_mutex_unlock(mutex) \
    pthread_mutex_unlock(mutex)
#define ort_pool_mutex_destroy(mutex) \
    pthread_mutex_destroy(mutex)
#define ort_pool_cond_init(cond) \
    pthread_cond_init(cond, NULL)
#define ort_pool_cond_wait(cond, mutex) \
    pthread_cond_wait(cond, mutex)
#define ort_pool_cond_signal(cond) \
    pthread_cond_signal(cond)
#define ort_pool_cond_broadcast(cond) \
    pthread_cond_broadcast(cond)
#define ort_pool_cond_destroy(cond) \
    pthread_cond_destroy(cond)
#endif

typedef struct _ort_task_t {
    ort_task_func_t func;
    void *arg;
    size_t count;
    volatile size_t next;
    volatile size_t completed;
    ort_mutex_t mutex;
    ort_cond_t cond;
    volatile int done;
    volatile int refcount;
} ort_task_t;

typedef struct _ort_pool_t {
    ort_thread_t *threads;
    size_t     size;
    ort_task_t *task;
    ort_mutex_t mutex;
    ort_cond_t cond;
    int stop;
} ort_pool_t;

ORT_POOL_LOCAL ort_pool_t ort_maths_pool;

void ort_pool_binary_worker(void *arg, size_t index, size_t count) {
    ort_pool_binary_ctx_t *ctx =
        (ort_pool_binary_ctx_t *)arg;

    size_t chunk = ctx->layout.chunk;
    size_t start = index * chunk;
    size_t end   = start + chunk;

    if (end > ctx->layout.total)
        end = ctx->layout.total;
    
    size_t n = end - start;
    if (n > 0) {
        ctx->op(
            (char*)ctx->result    + start * ctx->layout.element,
            (char*)ctx->a         + start * ctx->layout.element,
            (char*)ctx->b         + start * ctx->layout.element,
            n
        );
    }
}

void ort_pool_unary_worker(void *arg, size_t index, size_t count) {
    ort_pool_unary_ctx_t *ctx =
        (ort_pool_unary_ctx_t *)arg;

    size_t chunk = ctx->layout.chunk;
    size_t start = index * chunk;
    size_t end   = start + chunk;

    if (end > ctx->layout.total)
        end = ctx->layout.total;
    
    size_t n = end - start;

    if (n > 0) {
        ctx->op(
            (char*)ctx->result    + start * ctx->layout.element,
            (char*)ctx->a         + start * ctx->layout.element,
            n
        );
    }
}

void ort_pool_scalar_worker(void *arg, size_t index, size_t count) {
    ort_pool_scalar_ctx_t *ctx =
        (ort_pool_scalar_ctx_t *)arg;
    size_t chunk = ctx->layout.chunk;
    size_t start = index * chunk;
    size_t end   = start + chunk;

    if (end > ctx->layout.total)
        end = ctx->layout.total;
    
    size_t n = end - start;

    if (n > 0) {
        ctx->op(
            (char*)ctx->result    + start * ctx->layout.element,
            (char*)ctx->a         + start * ctx->layout.element,
            ctx->b,
            n
        );
    }
}

void ort_pool_slow_binary_worker(void *arg, size_t index, size_t count) {
    ort_pool_slow_binary_ctx_t *ctx =
        (ort_pool_slow_binary_ctx_t *)arg;
    size_t chunk = ctx->layout.chunk;
    size_t start = index * chunk;
    size_t end = start + chunk;
    
    if (end > ctx->layout.total)
        end = ctx->layout.total;
    size_t n = end - start;
    
    if (n == 0)
        return;

    int64_t *out_indices = NULL, *a_indices = NULL, *b_indices = NULL;
    zend_bool use_stack = (ctx->result_dimensions <= 8);
    int64_t out_indices_stack[8], a_indices_stack[8], b_indices_stack[8];
    
    if (use_stack) {
        out_indices = out_indices_stack;
        a_indices = a_indices_stack;
        b_indices = b_indices_stack;
    } else {
        out_indices = emalloc(
            ctx->result_dimensions * sizeof(int64_t));
        a_indices = emalloc(
            ctx->result_dimensions * sizeof(int64_t));
        b_indices = emalloc(
            ctx->result_dimensions * sizeof(int64_t));
    }

    char a_buf[16], b_buf[16];
    for (size_t flat = start; flat < end; flat++) {
        ort_math_result_multi(flat, ctx->result_shape, ctx->result_dimensions, out_indices);
        for (size_t i = 0; i < ctx->result_dimensions; i++) {
            a_indices[i] =
                (i < ctx->a_dim_offset) ? 0 :
                    (ctx->a_shape[i - ctx->a_dim_offset] == 
                        1 ? 0 : out_indices[i]);
            b_indices[i] =
                (i < ctx->b_dim_offset) ? 0 :
                    (ctx->b_shape[i - ctx->b_dim_offset] == 
                        1 ? 0 : out_indices[i]);
        }

        zend_long a_flat = ort_math_result_flat(
            a_indices + ctx->a_dim_offset,
            ctx->a_shape, ctx->a_dimensions);
        zend_long b_flat = ort_math_result_flat(
            b_indices + ctx->b_dim_offset,
            ctx->b_shape, ctx->b_dimensions);
        
        void *res_ptr =
            (char*)ctx->result +
                flat * ctx->element_size;
        void *a_ptr =
            (char*)ctx->a +
                a_flat * ctx->a_element_size;
        void *b_ptr =
            (char*)ctx->b +
                b_flat * ctx->b_element_size;

        ort_math_cast_element(a_ptr, a_buf, ctx->a_type, ctx->result_type);
        ort_math_cast_element(b_ptr, b_buf, ctx->b_type, ctx->result_type);

        ctx->op(res_ptr, a_buf, b_buf, 1);
    }

    if (!use_stack) {
        efree(out_indices);
        efree(a_indices);
        efree(b_indices);
    }
}

static void *ort_pool_worker(void *arg) {
    ort_pool_t *pool = (ort_pool_t *)arg;
    while (1) {
        ort_pool_mutex_lock(&pool->mutex);
        while (!pool->task && !pool->stop) {
            ort_pool_cond_wait(&pool->cond, &pool->mutex);
        }
        if (pool->stop) {
            ort_pool_mutex_unlock(&pool->mutex);
            break;
        }
        ort_task_t *task = pool->task;
        php_ort_atomic_addref((uint32_t*)&task->refcount);
        ort_pool_mutex_unlock(&pool->mutex);

        while (1) {
#if defined(_WIN32)
            size_t index = InterlockedIncrement64((volatile LONGLONG*)&task->next) - 1;
#else
            size_t index = __sync_fetch_and_add(&task->next, 1);
#endif
            if (index >= task->count)
                break;
            task->func(task->arg, index, task->count);
#if defined(_WIN32)
            InterlockedIncrement64((volatile LONGLONG*)&task->completed);
#else
            __sync_fetch_and_add(&task->completed, 1);
#endif
        }

        ort_pool_mutex_lock(&task->mutex);
        if (task->completed == task->count) {
            ort_pool_cond_signal(&task->cond);
            // Wait for main thread to set done
            while (!task->done) {
                ort_pool_cond_wait(&task->cond, &task->mutex);
            }
        }
        ort_pool_mutex_unlock(&task->mutex);
        // Drop worker's reference
        if (php_ort_atomic_delref((uint32_t*)&task->refcount) == 0) {
#if defined(_WIN32)
            DeleteCriticalSection(&task->mutex);
            free(task);
#else
            pthread_mutex_destroy(&task->mutex);
            pthread_cond_destroy(&task->cond);
            free(task);
#endif
        }
    }
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static inline size_t ort_pool_cores_env() {
    const char *env = getenv("ORT_POOL_CORES");
    if (env) {
        char *endptr;
        long n = strtol(env, &endptr, 10);
        if (*endptr == '\0' && n > 0) {
            return (size_t)n;
        }
    }
    return 0; // No valid environment variable set
}

#if defined(_WIN32)
size_t ort_pool_cores(void) {
    size_t env = ort_pool_cores_env();
    if (env > 0) {
        return env;
    }

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);

    return (size_t)sysinfo.dwNumberOfProcessors;
}
#else
size_t ort_pool_cores(void) {
    size_t env = ort_pool_cores_env();
    if (env > 0) {
        return env;
    }

    long n = sysconf(
        _SC_NPROCESSORS_ONLN);
    return (n > 0) ? (size_t)n : 1;
}
#endif

int ort_pool_init(size_t size) {
    memset(&ort_maths_pool, 0, sizeof(ort_maths_pool));
    if (size == 0)
        size = ort_pool_cores();
    ort_maths_pool.size = size;
    ort_maths_pool.threads =
        (ort_thread_t*)calloc(size, sizeof(ort_thread_t));
    if (!ort_maths_pool.threads) {
        ort_maths_pool.size = 0;
        return FAILURE;
    }
    ort_pool_mutex_init(&ort_maths_pool.mutex);
    ort_pool_cond_init(&ort_maths_pool.cond);
    ort_maths_pool.stop = 0;
    for (size_t i = 0; i < size; ++i) {
#if defined(_WIN32)
        ort_maths_pool.threads[i] = CreateThread(
            NULL, 0, 
            ort_pool_worker, 
            &ort_maths_pool, 0, NULL);
#else
        pthread_create(
            &ort_maths_pool.threads[i], 
            NULL, 
            ort_pool_worker, 
            &ort_maths_pool);
#endif
    }
    return SUCCESS;
}

void ort_pool_destroy() {
    ort_pool_mutex_lock(&ort_maths_pool.mutex);
    ort_maths_pool.stop = 1;
    ort_pool_cond_broadcast(&ort_maths_pool.cond);
    ort_pool_mutex_unlock(&ort_maths_pool.mutex);

    for (size_t i = 0; i < ort_maths_pool.size; ++i) {
#if defined(_WIN32)
        WaitForSingleObject(ort_maths_pool.threads[i], INFINITE);
        CloseHandle(ort_maths_pool.threads[i]);
#else
        pthread_join(ort_maths_pool.threads[i], NULL);
#endif
    }

    ort_pool_mutex_destroy(&ort_maths_pool.mutex);
    ort_pool_cond_destroy(&ort_maths_pool.cond);
    free(ort_maths_pool.threads);
}

void ort_pool_submit(ort_task_func_t func, void *arg, size_t count) {
    ort_task_t *task = (ort_task_t*)calloc(1, sizeof(ort_task_t));
    ort_pool_mutex_init(&task->mutex);
    ort_pool_cond_init(&task->cond);
    task->func = func;
    task->arg = arg;
    task->count = count;
    task->next = 0;
    task->completed = 0;
    task->done = 0;
    task->refcount = 1 + ort_maths_pool.size; // main thread + all workers

    ort_pool_mutex_lock(&ort_maths_pool.mutex);
    ort_maths_pool.task = task;
    ort_pool_cond_broadcast(&ort_maths_pool.cond);
    ort_pool_mutex_unlock(&ort_maths_pool.mutex);

    ort_pool_mutex_lock(&task->mutex);
    while (task->completed < task->count) {
        ort_pool_cond_wait(&task->cond, &task->mutex);
    }
    // Signal workers that it's safe to exit
    task->done = 1;
    ort_pool_cond_broadcast(&task->cond);
    ort_pool_mutex_unlock(&task->mutex);

    // Drop main thread's reference
    if (php_ort_atomic_delref((uint32_t*)&task->refcount) == 0) {
#if defined(_WIN32)
        DeleteCriticalSection(&task->mutex);
        free(task);
#else
        pthread_mutex_destroy(&task->mutex);
        pthread_cond_destroy(&task->cond);
        free(task);
#endif
    }

    ort_pool_mutex_lock(&ort_maths_pool.mutex);
    ort_maths_pool.task = NULL;
    ort_pool_mutex_unlock(&ort_maths_pool.mutex);
}
