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

#if defined(_WIN32)
# include <windows.h>
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
# define _GNU_SOURCE
# include <pthread.h>
# include <sched.h>
# include <unistd.h>
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

#include "maths/cast.h"
#include "maths/core.h"
#include "maths/pool.h"
#include "maths/result.h"

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

ORT_TLS ort_pool_t __ort_pool;
ORT_TLS size_t     __ort_pool_cores = 0;

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

void ort_pool_reduce_tensor_worker(void *arg, size_t index, size_t count) {
    ort_pool_reduce_tensor_ctx_t *ctx =
        (ort_pool_reduce_tensor_ctx_t *)arg;
    size_t chunk = ctx->layout.chunk;
    size_t start = index * chunk;
    size_t end = start + chunk;

    if (end > ctx->layout.total)
        end = ctx->layout.total;
    
    size_t n = end - start;

    if (n > 0) {
        // Each thread reduces a chunk of the input, but for full reduction to scalar,
        // only one thread should do the reduction (to avoid race conditions).
        // For now, only thread 0 does the reduction.
        if (start == 0) {
            ctx->op(
                ctx->result, ctx->a, ctx->elements);
        }
    }
}

void ort_pool_reduce_axis_worker(void *arg, size_t index, size_t count) {
    ort_pool_reduce_axis_ctx_t *ctx =
        (ort_pool_reduce_axis_ctx_t *)arg;

    size_t chunk     = ctx->layout.chunk;
    size_t outer     = ctx->layout.outer;
    size_t inner     = ctx->layout.inner;
    size_t axis_size = ctx->layout.axis_size;
    size_t element   = ctx->layout.element;

    size_t start = index * chunk;
    size_t end = start + chunk;
    if (end > outer)
        end = outer;

    if (start < end) {
        void *result_ptr =
            (char*)ctx->result +
                start * inner * element;
        const void *a_ptr =
            (const char*)ctx->a + 
                start * axis_size * inner * element;
        ctx->op(
            result_ptr,
            a_ptr,
            end - start,
            axis_size,
            inner);
    }
}

void ort_pool_matmul_worker(void *arg, size_t index, size_t count) {
    ort_pool_matmul_ctx_t *ctx =
        (ort_pool_matmul_ctx_t *)arg;
    size_t chunk = ctx->layout.chunk;
    size_t start = index * chunk;
    size_t end = start + chunk;

    if (end > ctx->layout.total)
        end = ctx->layout.total;
    
        for (size_t batch = start; batch < end; ++batch) {
        void *result_ptr =
            (char*)ctx->result +
                batch * ctx->matrix_size_result * ctx->type_size;
        const void *a_ptr =
            (const char*)ctx->a + batch * ctx->matrix_size_a * ctx->type_size;
        const void *b_ptr =
            (const char*)ctx->b + batch * ctx->matrix_size_b * ctx->type_size;
        
        ctx->op(result_ptr, a_ptr, b_ptr, ctx->a_rows, ctx->a_cols, ctx->b_cols);
    }
}

void ort_pool_cast_worker(void *arg, size_t index, size_t count) {
    ort_pool_cast_ctx_t *ctx =
        (ort_pool_cast_ctx_t *)arg;

    size_t chunk = (ctx->count + count - 1) / count;
    size_t start = index * chunk;
    size_t end = start + chunk;

    if (end > ctx->count)
        end = ctx->count;

    size_t src_elem_size = php_ort_type_sizeof(ctx->src_type);
    size_t dst_elem_size = php_ort_type_sizeof(ctx->dst_type);

    const char *src =
        (const char *)ctx->src +
            start * src_elem_size;
    char *dst =
        (char *)ctx->dst +
            start * dst_elem_size;

    for (size_t i = 0; i < end - start; ++i) {
        ctx->op(
            src + i * src_elem_size,
            dst + i * dst_elem_size,
            ctx->src_type,
            ctx->dst_type
        );
    }
}

/*
 * Pin the current thread to the core it is currently running on.
 * This guarantees no migration, regardless of platform.
 */
static void ort_pool_pin(void) {
#ifdef _WIN32
    DWORD cpu =
        GetCurrentProcessorNumber();
    DWORD_PTR mask =
        ((DWORD_PTR)1) << cpu;
    SetThreadAffinityMask(
        GetCurrentThread(), mask);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    int cpu = sched_getcpu();
    if (cpu >= 0) {
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(
            pthread_self(), sizeof(set), &set);
    }
#endif
}

static void *ort_pool_worker(void *arg) {
    ort_pool_t *pool = (ort_pool_t *)arg;

    /* Pin this thread to the core it is currently running on */
    ort_pool_pin();

    /* Startup the allocator for this thread */
    PHP_MINIT_FUNCTION(ORT_ALLOC);

    /* Startup math library for this thread */
    ort_math_startup();

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

    /* Shutdown math library for this thread */
    ort_math_shutdown();

    /* Shutdown the allocator for this thread */
    PHP_MSHUTDOWN_FUNCTION(ORT_ALLOC);

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

size_t ort_pool_max(void) {
    return __ort_pool.size;
}

size_t ort_pool_cores(void) {
    if (__ort_pool_cores > 0) {
        return __ort_pool_cores;
    }

#if defined(_WIN32)
    __ort_pool_cores = ort_pool_cores_env();
    if (__ort_pool_cores > 0) {
        return __ort_pool_cores;
    }

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);

    __ort_pool_cores =
        (size_t)sysinfo.dwNumberOfProcessors;
#else
    __ort_pool_cores = ort_pool_cores_env();
    if (__ort_pool_cores > 0) {
        return __ort_pool_cores;
    }

    long n = sysconf(
        _SC_NPROCESSORS_ONLN);

    __ort_pool_cores =
        (n > 0) ? (size_t)n : 1;
#endif
    return __ort_pool_cores;
}

size_t ort_pool_scale(size_t cores) {
    if (cores > ort_pool_max()) {
        /** 
         We cannot scale beyond the limit of the numbers of cores 
        **/
        return 0;
    }

    size_t restore =
        __ort_pool_cores;
    __ort_pool_cores = cores;

    return restore;
}

int ort_pool_init(size_t size) {
    memset(&__ort_pool, 0, sizeof(__ort_pool));
    if (size == 0)
        size = ort_pool_cores();
    __ort_pool.size = size;
    __ort_pool.threads =
        (ort_thread_t*)calloc(size, sizeof(ort_thread_t));
    if (!__ort_pool.threads) {
        __ort_pool.size = 0;
        return FAILURE;
    }
    ort_pool_mutex_init(&__ort_pool.mutex);
    ort_pool_cond_init(&__ort_pool.cond);
    __ort_pool.stop = 0;

    for (size_t i = 0; i < size; ++i) {
#if defined(_WIN32)
        __ort_pool.threads[i] = CreateThread(
            NULL, 0,
            (LPTHREAD_START_ROUTINE)ort_pool_worker,
            &__ort_pool, 0, NULL);
#else
        pthread_create(
            &__ort_pool.threads[i],
            NULL,
            ort_pool_worker,
            &__ort_pool);
#endif
    }
    return SUCCESS;
}

void ort_pool_destroy() {
    ort_pool_mutex_lock(&__ort_pool.mutex);
    __ort_pool.stop = 1;
    ort_pool_cond_broadcast(&__ort_pool.cond);
    ort_pool_mutex_unlock(&__ort_pool.mutex);

    for (size_t i = 0; i < __ort_pool.size; ++i) {
#if defined(_WIN32)
        WaitForSingleObject(__ort_pool.threads[i], INFINITE);
        CloseHandle(__ort_pool.threads[i]);
#else
        pthread_join(__ort_pool.threads[i], NULL);
#endif
    }

    ort_pool_mutex_destroy(&__ort_pool.mutex);
    ort_pool_cond_destroy(&__ort_pool.cond);
    free(__ort_pool.threads);
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
    task->refcount = 1 + __ort_pool.size; // main thread + all workers

    ort_pool_mutex_lock(&__ort_pool.mutex);
    __ort_pool.task = task;
    ort_pool_cond_broadcast(&__ort_pool.cond);
    ort_pool_mutex_unlock(&__ort_pool.mutex);

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

    ort_pool_mutex_lock(&__ort_pool.mutex);
    __ort_pool.task = NULL;
    ort_pool_mutex_unlock(&__ort_pool.mutex);
}
