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
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif

#include "maths/cast.h"
#include "maths/core.h"
#include "maths/pool.h"
#include "maths/result.h"

#ifdef HAVE_ORT_POOL
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
#define ort_pool_thread_self() \
   GetCurrentThread()
#else
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
#define ort_pool_thread_self() \
   pthread_self()
#endif
#endif

#define _ORT_POOL_PAD(TYPE, CACHE, SIZE) \
    TYPE pad[CACHE - (SIZE)]

typedef struct _ort_task_t {
    volatile ort_task_func_t func;
    volatile void *arg;
    size_t count;
    volatile int completed;
    /* 
    This wastes a few bytes, but eliminates the possibility of false sharing:
    
      Essentially, because this structure doesn't fill a cache line, we padd to that width.
      This means that no two adjacent structures share a cache line, which avoids false sharing.

    Exhaustive explanation: https://en.wikipedia.org/wiki/False_sharing

    TL;DR: It's worth a few bytes ...
    */
    _ORT_POOL_PAD(
        uint8_t,
        64,
        sizeof(ort_task_func_t) +
        sizeof(void*)           +
        sizeof(size_t)          +
        sizeof(int));
} ort_task_t;

typedef struct _ort_pool_arg_t ort_pool_arg_t;

typedef struct _ort_pool_t {
#ifdef HAVE_ORT_POOL
   ort_thread_t *threads;
#ifdef _WIN32
    /**
     On windows we need identifiers from the scheduler
     because the handles are not directly comparable
    */
    DWORD* idx;
#endif
   ort_mutex_t mutex;
   ort_cond_t cond;
   ort_pool_arg_t *arg;
#endif
   size_t     size;
   ort_task_t *slots;
   volatile size_t activated;
   volatile int stop;
} ort_pool_t;

typedef struct _ort_pool_arg_t {
    ort_pool_t *pool;
    size_t index;
} _ort_pool_arg_t;

ORT_TLS ort_pool_t __ort_pool;
ORT_TLS size_t     __ort_pool_cores = 0;
ORT_TLS size_t     __ort_pool_threshold = 0;

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

   size_t chunk   = ctx->layout.chunk;
   size_t element = ctx->layout.element;
   size_t total   = ctx->layout.total;

   size_t start = index * chunk;
   size_t end   = start + chunk;
   if (end > total)
       end = total;

   if (start < end) {
       void *result_ptr =
           (char*)ctx->result + start * element;
       const void *a_ptr =
           (const char*)ctx->a + start * element;
       ctx->op(
           result_ptr,
           a_ptr,
           ctx->input_shape,
           ctx->input_dims,
           ctx->output_shape,
           ctx->output_dims,
           ctx->axis);
   }
}

void ort_pool_transform_axis_worker(void *arg, size_t index, size_t count) {
   ort_pool_transform_axis_ctx_t *ctx =
       (ort_pool_transform_axis_ctx_t *)arg;

   size_t chunk   = ctx->layout.chunk;
   size_t element = ctx->layout.element;
   size_t total   = ctx->layout.total;

   size_t start = index * chunk;
   size_t end   = start + chunk;
   if (end > total)
       end = total;

   if (start < end) {
       void *result_ptr =
           (char*)ctx->result + start * element;
       const void *a_ptr =
           (const char*)ctx->a + start * element;
       ctx->op(
           result_ptr,
           a_ptr,
           ctx->shape,
           ctx->dims,
           ctx->axis);
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

   // Each thread computes a range of output rows
   for (size_t row = start; row < end; ++row) {
       void *result_ptr =
           (char*)ctx->result +
               row * ctx->b_cols * ctx->type_size;
       const void *a_ptr =
           (const char*)ctx->a +
               row * ctx->a_cols * ctx->type_size;
       const void *b_ptr = ctx->b;

       ctx->op(
           result_ptr,
           a_ptr, b_ptr,
           ctx->a_cols, ctx->b_cols);
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

#ifdef HAVE_ORT_POOL
/*
* Pin the current thread to the target core.
*
* We don't leave it to the scheduler to decide how to distribute threads, beause this leads to non-determinisic behavior.
* 
* Instead, we pin each thread to the target core (determined by topology or ORT_SCALE_CORES), this distributes the threads evenly.
*/
static void ort_pool_pin(size_t index) {
   /* We set affinity and wait for the scheduler to migrate the thread (if necessary) */
#ifdef _WIN32
   DWORD_PTR mask =
       ((DWORD_PTR)1) << (DWORD)index;
   SetThreadAffinityMask(
       GetCurrentThread(), mask);
   /* The windows scheduler can ignore us ...*/
   uint32_t max = 100;
   while (GetCurrentProcessorNumber() != (DWORD) index) {
       Sleep(0);
       if (--max == 0) {
           /* So we ignore it right back ... */

           /*
             This is a platform limitation, the scheduler does not
             have to migrate the thread.

             We stick around a little, to give it a chance, then move on
           */
           break;
       }
   }
#elif defined(__APPLE__) || defined(__MACH__)
   /*
   * macOS doesn't support strict CPU affinity. Threads are not pinned to specific cores.
   * This is safe because the ISA (including NEON) is consistent across all cores.
   *
   * However, we can provide a performance hint to the scheduler using Quality of Service (QoS)
   * to encourage it to run this thread on a high-performance P-core.
   *
   * The scheduler still has final say ...
   */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);

    (void)index; /* Avoid unused parameter warning */
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(index, &set);
    pthread_setaffinity_np(
        pthread_self(), sizeof(set), &set);
    while (sched_getcpu() != index) {
        sched_yield();
    }
#endif
    /* The thread is now executing on the target core, as determined by topology or ORT_SCALE_CORES */
}
#else
static inline void ort_pool_pin(size_t index) {
    /* No pooling, so no pinning */
    (void)index;
}
#endif

#ifdef HAVE_ORT_POOL
typedef enum _ort_pool_worker_status_t {
    ORT_POOL_WORKER_CONTINUE,
    ORT_POOL_WORKER_SIGNAL
} ort_pool_worker_status_t;

static inline ort_pool_worker_status_t
    ort_pool_worker_complete(
        ort_pool_t *pool,
        ort_task_t *task) {
#if defined(_WIN32)
    InterlockedExchange(
        (volatile LONG*)&task->completed, 1);
#else
    __sync_lock_test_and_set(
        &task->completed, 1);
#endif

    if (
#if defined(_WIN32)
    InterlockedDecrement64(
        (volatile LONGLONG*)&pool->activated) == 0
#else
    __sync_fetch_and_sub(&pool->activated, 1) == 1
#endif
    ) {
        return ORT_POOL_WORKER_SIGNAL;
    }
    return ORT_POOL_WORKER_CONTINUE;
}

static void *ort_pool_worker(void *_arg) {
   ort_pool_arg_t *arg =
    (ort_pool_arg_t *) _arg;
   ort_pool_t *pool = arg->pool;
   size_t idx = arg->index;

   /* Pin this thread to the target core */
   ort_pool_pin(idx);
   /* Startup the allocator for this thread */
   ort_alloc_startup();
   /* Startup math library for this thread */
   ort_math_startup();

   /* Execute activate routine possible installed by backend */
   ort_alloc_activate();

   while (1) {
       ort_pool_mutex_lock(&pool->mutex);
       while (!pool->activated && !pool->stop) {
           ort_pool_cond_wait(&pool->cond, &pool->mutex);
       }
       if (pool->stop) {
           ort_pool_mutex_unlock(&pool->mutex);
           break;
       }

       /* Check our task slot */
       ort_task_t *task = &pool->slots[idx];

       if (task->func && !task->completed) {
           ort_pool_mutex_unlock(&pool->mutex);

           /* Execute the task assigned to this thread */
           task->func((void*)task->arg, idx, task->count);

           /* Mark task completed */
           if (ort_pool_worker_complete(pool, task) ==
                   ORT_POOL_WORKER_SIGNAL) {
                /* Last completer must signal to submitter */
                ort_pool_mutex_lock(&pool->mutex);
                ort_pool_cond_broadcast(
                    &pool->cond);
                ort_pool_mutex_unlock(&pool->mutex);
           }
       } else {
           ort_pool_mutex_unlock(&pool->mutex);
       }
   }

   /* Execute deactivate routine possible installed by backend */
   ort_alloc_deactivate();

   /* Shutdown math library for this thread */
   ort_math_shutdown();

   /* Shutdown the allocator for this thread */
   ort_alloc_shutdown();

#if defined(_WIN32)
   return 0;
#else
   return NULL;
#endif
}
#endif

static inline size_t ort_pool_cores_env(void) {
   const char *env = getenv("ORT_SCALE_CORES");
   if (env) {
       char *endptr;
       long n = strtol(env, &endptr, 10);
       if (*endptr == '\0' && n > 0) {
           return (size_t)n;
       }
   }
   return 0; // No valid environment variable set
}

static inline size_t ort_pool_threshold_env(void) {
   const char *env = getenv("ORT_SCALE_THRESHOLD");
   if (env) {
       char *endptr;
       long n = strtol(env, &endptr, 10);
       if (*endptr == '\0' && n > 0) {
           return (size_t)n;
       }
   }
   return 0; // No valid environment variable set
}

#ifdef HAVE_ORT_POOL
size_t ort_pool_max(void) {
   return __ort_pool.size;
}
#else
size_t ort_pool_max(void) {
   return 1;
}
#endif

#ifdef HAVE_ORT_POOL
/* {{{ Retrieve the actual number of cores available (regardless of env) */
static zend_always_inline size_t ort_pool_threads(void) {
   size_t threads = 0;

#if defined(_WIN32)
   SYSTEM_INFO sysinfo;
   GetSystemInfo(&sysinfo);

   threads = (size_t)
       sysinfo.dwNumberOfProcessors;
#else
   threads = (size_t) sysconf(
       _SC_NPROCESSORS_ONLN);
#endif

#if !defined(__APPLE__) && !defined(__MACH__)
# ifdef ORT_BACKEND_CPU_ENABLED
    /**
    * Here we adjust the default number of threads by the number of threads
    * that would be guarded.
    *
    * We do this to stabilize performance, so that we don't create more workers
    * than there are cores that would allow the installation of the backend. Thus all
    * worker threads will be using the backend for computation, not a mixture of frontend
    * and backend that would result if we don't make this adjustment.
    * 
    * @see backend/guard.h
    */
    size_t guarded = 0;
    for (size_t thread = 0; thread < threads; thread++) {
        /* migrate across the package */
        ort_pool_pin(thread);
        /* check if the current backend would be guarded on this core */
        if (ORT_MATH_BACKEND_GUARD(ORT_BACKEND_CPU_TYPE)){
            guarded++;
        }
    }
    threads -= guarded;

    /* We must migrate back to the start */
    ort_pool_pin(0);
# endif
#endif

   return threads;
} /* }}} */
#else
static zend_always_inline size_t ort_pool_threads(void) {
   return 1; // No pooling, so only one thread
}
#endif

#ifdef HAVE_ORT_POOL
size_t ort_pool_cores(void) {
   if (__ort_pool_cores > 0) {
       return __ort_pool_cores;
   }

   __ort_pool_cores = ort_pool_cores_env();
   if (__ort_pool_cores > 0) {
       /*
        We prohibit over subscription
       */
       size_t check = ort_pool_threads();

       /* We silently clamp this, and will document this behavior; It is too early in startup 
           to raise a graceful error in all cases. */
       if (__ort_pool_cores > check) {
           __ort_pool_cores = check;
       }

       return __ort_pool_cores;
   }

   __ort_pool_cores =
       ort_pool_threads();

   return __ort_pool_cores;
}

size_t ort_pool_threshold(void) {
   if (__ort_pool_threshold > 0) {
       return __ort_pool_threshold;
   }

   __ort_pool_threshold = ort_pool_threshold_env();
   if (__ort_pool_threshold > 0) {
       return __ort_pool_threshold;
   }

   __ort_pool_threshold = ORT_SCALE_THRESHOLD;

   return __ort_pool_threshold;
}
#else
size_t ort_pool_cores(void) {
   return 1; // No pooling, so only one thread
}
size_t ort_pool_threshold(void) {
   return ORT_SCALE_THRESHOLD; // No pooling, so use default threshold
}
#endif

ort_pool_scale_t ort_pool_scale(ort_pool_scale_t* scale) {
   if (scale->kind & ORT_POOL_SCALE_ERROR) {
       /* undefined behavior */
       return *scale;
   }

   ort_pool_scale_t restore =
       (ort_pool_scale_t) {
       .kind = ORT_POOL_SCALE_CORES |
               ORT_POOL_SCALE_THRESHOLD,
       .cores     = __ort_pool_cores,
       .threshold = __ort_pool_threshold
   };

   if (scale->cores <= 0 || scale->cores > ort_pool_max()) {
       /** 
        We cannot scale below zero or beyond the limit of the numbers of cores
        This is probably a programming error
       **/
       restore.kind |= 
           ORT_POOL_SCALE_ERROR;
       return restore;
   }

   /* Perform threshold adjustment only if requested */
   if (scale->kind & ORT_POOL_SCALE_THRESHOLD) {
       if (scale->threshold <= 0) {
           /**
            * This must be a programming error
            */
           restore.kind |=
               ORT_POOL_SCALE_ERROR;
           return restore;
       }

       __ort_pool_threshold = (size_t) scale->threshold;
   }

   /*
   We always scale cores
   */
   __ort_pool_cores = (size_t) scale->cores;

   return restore;
}

#ifdef HAVE_ORT_POOL
int ort_pool_init(size_t size) {
   ort_pool_pin(0);

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
#ifdef _WIN32
   __ort_pool.idx = calloc(sizeof(DWORD), size);
   if (!__ort_pool.idx) {
       __ort_pool.size = 0;
       return FAILURE;
   }
#endif
   __ort_pool.slots =
       (ort_task_t*)calloc(size, sizeof(ort_task_t));
   if (!__ort_pool.slots) {
       free(__ort_pool.threads);
       __ort_pool.size = 0;
       return FAILURE;
   }
#ifdef HAVE_ORT_POOL
   __ort_pool.arg = 
       (ort_pool_arg_t*)calloc(size, sizeof(ort_pool_arg_t));
   if (!__ort_pool.arg) {
       free(__ort_pool.threads);
       free(__ort_pool.slots);
       __ort_pool.size = 0;
       return FAILURE;
   }
#endif
   ort_pool_mutex_init(&__ort_pool.mutex);
   ort_pool_cond_init(&__ort_pool.cond);
   __ort_pool.stop = 0;

   for (size_t i = 0; i < size; ++i) {
#ifdef HAVE_ORT_POOL
       __ort_pool.arg[i].pool = &__ort_pool;
       __ort_pool.arg[i].index = i;
#endif
#if defined(_WIN32)
       __ort_pool.threads[i] = CreateThread(
           NULL, 0,
           (LPTHREAD_START_ROUTINE)ort_pool_worker,
           &__ort_pool.arg[i], 0, NULL);
        __ort_pool.idx[i] = GetThreadId(__ort_pool.threads[i]);
#else
        pthread_create(
            &__ort_pool.threads[i],
            NULL,
            ort_pool_worker,
            &__ort_pool.arg[i]);
#endif
   }
   return SUCCESS;
}
#else
int ort_pool_init(size_t size) {
   /* No-op if pooling is not enabled */
   (void)size; // Avoid unused parameter warning
   return SUCCESS;
}
#endif

#ifdef HAVE_ORT_POOL
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

#ifdef _WIN32
   free(__ort_pool.idx);
#endif

   ort_pool_mutex_destroy(&__ort_pool.mutex);
   ort_pool_cond_destroy(&__ort_pool.cond);
   free(__ort_pool.threads);
   free(__ort_pool.slots);
   free(__ort_pool.arg);
}
#else
void ort_pool_destroy() {
    /* No-op if pooling is not enabled */
}
#endif

#ifdef HAVE_ORT_POOL
void ort_pool_submit(ort_task_func_t func, void *arg, size_t count) {
   /* Prepare task slots for each thread */
   ort_pool_mutex_lock(&__ort_pool.mutex);

   /* Clear all task slots */
   memset(__ort_pool.slots, 0, __ort_pool.size * sizeof(ort_task_t));

   /* Distribute work across threads */
   __ort_pool.activated = 0;
   for (size_t i = 0; i < __ort_pool.size; ++i) {
       if (i < count) {
        __ort_pool.slots[i].func      = func;
        __ort_pool.slots[i].arg       = arg;
        __ort_pool.slots[i].count     = count;
        __ort_pool.slots[i].completed = 0;
        __ort_pool.activated++;
       } else {
        __ort_pool.slots[i].func      = NULL;
        __ort_pool.slots[i].arg       = NULL;
        __ort_pool.slots[i].count     = 0;
        __ort_pool.slots[i].completed = 1;
       }
   }
   
   /* Wake all threads */
   ort_pool_cond_broadcast(&__ort_pool.cond);
   ort_pool_mutex_unlock(&__ort_pool.mutex);

   /* Wait for completion */
   ort_pool_mutex_lock(&__ort_pool.mutex);
   while (__ort_pool.activated > 0) {
      ort_pool_cond_wait(
         &__ort_pool.cond, &__ort_pool.mutex);
   }

   /* Clearup slots (dont do strange things during spurious wakeups) */
   for (size_t i = 0; i < __ort_pool.size && i < count; ++i) {
       __ort_pool.slots[i].func = NULL;
       __ort_pool.slots[i].arg = NULL;
       __ort_pool.slots[i].completed = 1;
   }

   ort_pool_mutex_unlock(&__ort_pool.mutex);
}
#else
void ort_pool_submit(ort_task_func_t func, void *arg, size_t count) {
   func(arg, 0, count);
}
#endif
