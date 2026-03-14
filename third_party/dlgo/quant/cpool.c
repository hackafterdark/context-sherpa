//go:build amd64 && cgo

// cpool.c — C-level thread pool for LLM matmul parallelization.
// Workers spin-wait with _mm_pause() for sub-µs dispatch latency.
// Pool is lazily initialized on first matmul call.

#pragma GCC target("avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")
#pragma GCC optimize("O3")

#include <windows.h>
#include <process.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>

#define CPOOL_MAX_WORKERS 64

// From simd_qq_dot.c
float qq_dot_q4_0_q8_0(const uint8_t* restrict xb, const uint8_t* restrict yb, int n);
float qq_dot_q8_0_q8_0(const uint8_t* restrict xb, const uint8_t* restrict yb, int n);
float qq_dot_q2_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n);
float qq_dot_q3_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n);
float qq_dot_q4_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n);
float qq_dot_q5_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n);
float qq_dot_q6_K_q8_K(const uint8_t* restrict xb, const uint8_t* restrict yb, int n);

// From simd_dot.c
float vec_dot_f16(const uint8_t* data, const float* x, int n);
float vec_dot_q4_0(const uint8_t* data, const float* x, int n);
float vec_dot_q4_1(const uint8_t* data, const float* x, int n);
float vec_dot_q5_0(const uint8_t* data, const float* x, int n);
float vec_dot_q5_1(const uint8_t* data, const float* x, int n);
float vec_dot_q8_0(const uint8_t* data, const float* x, int n);
float vec_dot_q2_k(const uint8_t* data, const float* x, int n);
float vec_dot_q3_k(const uint8_t* data, const float* x, int n);
float vec_dot_q4_k(const uint8_t* data, const float* x, int n);
float vec_dot_q5_k(const uint8_t* data, const float* x, int n);
float vec_dot_q6_k(const uint8_t* data, const float* x, int n);

void quantize_for_type(const float* x, uint8_t* out, uint32_t w_type, int n);

// ── Task ─────────────────────────────────────────────────────
typedef struct {
    const uint8_t* w_data;
    uint32_t       w_type;
    const void*    input;
    int            cols;
    float*         out;
    int            bpr;
    int            start_row;
    int            end_row;
    int            use_qq;
    // batch GEMM fields
    int            is_batch;
    int            n_inputs;
    int            q8_stride;
    int            out_stride;
} cpool_task_t;

typedef struct {
    _Alignas(64) atomic_int has_task;
    cpool_task_t            task;
    atomic_int*             done;
} cpool_worker_t;

static _Alignas(64) cpool_worker_t cpool_workers[CPOOL_MAX_WORKERS];
static int cpool_nworkers = 0;
static atomic_int cpool_alive = 0;
static _Alignas(64) atomic_int cpool_active = 0;
static HANDLE cpool_wake_event = NULL;

// ── Row computation ──────────────────────────────────────────
static inline float qq_row(const uint8_t* row, uint32_t t,
                           const uint8_t* q, int n) {
    switch (t) {
    case 2:  return qq_dot_q4_0_q8_0(row, q, n);
    case 8:  return qq_dot_q8_0_q8_0(row, q, n);
    case 10: return qq_dot_q2_K_q8_K(row, q, n);
    case 11: return qq_dot_q3_K_q8_K(row, q, n);
    case 12: return qq_dot_q4_K_q8_K(row, q, n);
    case 13: return qq_dot_q5_K_q8_K(row, q, n);
    case 14: return qq_dot_q6_K_q8_K(row, q, n);
    default: return 0;
    }
}

static inline float fused_row(const uint8_t* row, uint32_t t,
                               const float* x, int n) {
    switch (t) {
    case 1:  return vec_dot_f16(row, x, n);
    case 2:  return vec_dot_q4_0(row, x, n);
    case 3:  return vec_dot_q4_1(row, x, n);
    case 6:  return vec_dot_q5_0(row, x, n);
    case 7:  return vec_dot_q5_1(row, x, n);
    case 8:  return vec_dot_q8_0(row, x, n);
    case 10: return vec_dot_q2_k(row, x, n);
    case 11: return vec_dot_q3_k(row, x, n);
    case 12: return vec_dot_q4_k(row, x, n);
    case 13: return vec_dot_q5_k(row, x, n);
    case 14: return vec_dot_q6_k(row, x, n);
    default: return 0;
    }
}

static void run_task(cpool_task_t* t) {
    if (t->is_batch) {
        const uint8_t* q8_base = (const uint8_t*)t->input;
        int nrows = t->end_row - t->start_row;
        const uint8_t* w_start = t->w_data + (size_t)t->start_row * t->bpr;
        for (int p = 0; p < t->n_inputs; p++) {
            const uint8_t* q8 = q8_base + (size_t)p * t->q8_stride;
            float* out = t->out + (size_t)p * t->out_stride + t->start_row;
            for (int r = 0; r < nrows; r++) {
                out[r] = qq_row(w_start + (size_t)r * t->bpr, t->w_type, q8, t->cols);
            }
        }
    } else if (t->use_qq) {
        const uint8_t* q = (const uint8_t*)t->input;
        for (int r = t->start_row; r < t->end_row; r++) {
            t->out[r] = qq_row(t->w_data + (size_t)r * t->bpr, t->w_type, q, t->cols);
        }
    } else {
        const float* x = (const float*)t->input;
        for (int r = t->start_row; r < t->end_row; r++) {
            t->out[r] = fused_row(t->w_data + (size_t)r * t->bpr, t->w_type, x, t->cols);
        }
    }
}

// ── Worker: hybrid spin-wait / sleep ─────────────────────────
static unsigned __stdcall worker_fn(void* arg) {
    cpool_worker_t* w = (cpool_worker_t*)arg;
    while (atomic_load_explicit(&cpool_alive, memory_order_relaxed)) {
        if (atomic_load_explicit(&w->has_task, memory_order_acquire)) {
            run_task(&w->task);
            atomic_store_explicit(&w->has_task, 0, memory_order_relaxed);
            atomic_fetch_sub_explicit(w->done, 1, memory_order_release);
        } else if (atomic_load_explicit(&cpool_active, memory_order_relaxed)) {
            _mm_pause();
        } else {
            WaitForSingleObject(cpool_wake_event, 50);
        }
    }
    return 0;
}

// ── Lifecycle ────────────────────────────────────────────────
void cpool_init(int n) {
    if (n > CPOOL_MAX_WORKERS) n = CPOOL_MAX_WORKERS;
    if (n < 1) n = 1;
    cpool_nworkers = n;
    atomic_store(&cpool_alive, 1);
    atomic_store(&cpool_active, 0);
    cpool_wake_event = CreateEventA(NULL, TRUE, FALSE, NULL);
    memset(cpool_workers, 0, sizeof(cpool_workers));
    for (int i = 0; i < n; i++) {
        atomic_store(&cpool_workers[i].has_task, 0);
        HANDLE h = (HANDLE)_beginthreadex(NULL, 0, worker_fn,
                                          &cpool_workers[i], 0, NULL);
        if (h) CloseHandle(h);
    }
}

void cpool_shutdown(void) {
    atomic_store(&cpool_alive, 0);
    Sleep(20);
}

int cpool_workers_count(void) { return cpool_nworkers; }

// ── Activate/deactivate pool ─────────────────────────────────
static void cpool_activate(void) {
    atomic_store_explicit(&cpool_active, 1, memory_order_release);
    SetEvent(cpool_wake_event);
}

static void cpool_deactivate(void) {
    atomic_store_explicit(&cpool_active, 0, memory_order_release);
}

// ── Core parallel dispatch ───────────────────────────────────
static void dispatch_rows(
    const uint8_t* w_data, uint32_t w_type,
    const void* input, int cols, float* out,
    int nrows, int bpr, int use_qq)
{
    if (nrows <= 0) return;
    int nw = cpool_nworkers;
    if (nw <= 0) {
        cpool_task_t t = {w_data, w_type, input, cols, out, bpr, 0, nrows, use_qq, 0, 0, 0, 0};
        run_task(&t);
        return;
    }
    if (nw > nrows) nw = nrows;

    cpool_activate();

    atomic_int done;
    atomic_store_explicit(&done, nw, memory_order_relaxed);

    int chunk = nrows / nw;
    int extra = nrows % nw;
    int row = 0;

    for (int i = 0; i < nw; i++) {
        int cnt = chunk + (i < extra ? 1 : 0);
        cpool_worker_t* w = &cpool_workers[i];
        w->task = (cpool_task_t){w_data, w_type, input, cols, out, bpr, row, row + cnt, use_qq, 0, 0, 0, 0};
        w->done = &done;
        row += cnt;
        atomic_store_explicit(&w->has_task, 1, memory_order_release);
    }
    while (atomic_load_explicit(&done, memory_order_acquire) > 0)
        _mm_pause();

    cpool_deactivate();
}

// ── Batch dispatch for multiple positions ────────────────────
static void dispatch_batch_rows(
    const uint8_t* w_data, uint32_t w_type,
    const uint8_t* q8_flat, int q8_stride, int n_inputs,
    int cols, float* out_flat, int nrows, int out_stride, int bpr)
{
    if (nrows <= 0) return;
    int nw = cpool_nworkers;
    if (nw <= 0) {
        for (int p = 0; p < n_inputs; p++) {
            const uint8_t* q8 = q8_flat + (size_t)p * q8_stride;
            float* out = out_flat + (size_t)p * out_stride;
            for (int r = 0; r < nrows; r++)
                out[r] = qq_row(w_data + (size_t)r * bpr, w_type, q8, cols);
        }
        return;
    }
    if (nw > nrows) nw = nrows;

    atomic_int done;
    atomic_store_explicit(&done, nw, memory_order_relaxed);

    int chunk = nrows / nw;
    int extra = nrows % nw;
    int row = 0;

    for (int i = 0; i < nw; i++) {
        int cnt = chunk + (i < extra ? 1 : 0);
        cpool_worker_t* w = &cpool_workers[i];
        w->task = (cpool_task_t){w_data, w_type, q8_flat, cols, out_flat, bpr,
                                  row, row + cnt, 1, 1, n_inputs, q8_stride, out_stride};
        w->done = &done;
        row += cnt;
        atomic_store_explicit(&w->has_task, 1, memory_order_release);
    }
    while (atomic_load_explicit(&done, memory_order_acquire) > 0)
        _mm_pause();
}

// ── Public API ───────────────────────────────────────────────
void cpool_qq_matvec(
    const uint8_t* w_data, uint32_t w_type,
    const float* x, int cols,
    float* out, int nrows, int bpr,
    uint8_t* q8_buf)
{
    quantize_for_type(x, q8_buf, w_type, cols);
    dispatch_rows(w_data, w_type, q8_buf, cols, out, nrows, bpr, 1);
}

void cpool_fused_matvec(
    const uint8_t* w_data, uint32_t w_type,
    const float* x, int cols,
    float* out, int nrows, int bpr)
{
    dispatch_rows(w_data, w_type, x, cols, out, nrows, bpr, 0);
}

void cpool_qq_batch_gemm(
    const uint8_t* w_data, uint32_t w_type,
    const uint8_t* q8_flat, int q8_stride, int n_inputs,
    int cols, float* out_flat, int nrows, int out_stride, int bpr)
{
    cpool_activate();
    dispatch_batch_rows(w_data, w_type, q8_flat, q8_stride, n_inputs,
                        cols, out_flat, nrows, out_stride, bpr);
    cpool_deactivate();
}

void cpool_qq_dual_batch_gemm(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const uint8_t* q8_flat, int q8_stride, int n_inputs,
    int cols, int out_stride1, int out_stride2)
{
    cpool_activate();
    dispatch_batch_rows(w1, t1, q8_flat, q8_stride, n_inputs, cols, o1, r1, out_stride1, bpr1);
    dispatch_batch_rows(w2, t2, q8_flat, q8_stride, n_inputs, cols, o2, r2, out_stride2, bpr2);
    cpool_deactivate();
}

void cpool_qq_triple_batch_gemm(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const uint8_t* w3, uint32_t t3, int r3, int bpr3, float* o3,
    const uint8_t* q8_flat, int q8_stride, int n_inputs,
    int cols, int out_stride1, int out_stride2, int out_stride3)
{
    cpool_activate();
    dispatch_batch_rows(w1, t1, q8_flat, q8_stride, n_inputs, cols, o1, r1, out_stride1, bpr1);
    dispatch_batch_rows(w2, t2, q8_flat, q8_stride, n_inputs, cols, o2, r2, out_stride2, bpr2);
    dispatch_batch_rows(w3, t3, q8_flat, q8_stride, n_inputs, cols, o3, r3, out_stride3, bpr3);
    cpool_deactivate();
}

void cpool_qq_dual_matvec(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const float* x, int cols, uint8_t* q8_buf)
{
    quantize_for_type(x, q8_buf, t1, cols);
    dispatch_rows(w1, t1, q8_buf, cols, o1, r1, bpr1, 1);
    dispatch_rows(w2, t2, q8_buf, cols, o2, r2, bpr2, 1);
}

void cpool_qq_triple_matvec(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const uint8_t* w3, uint32_t t3, int r3, int bpr3, float* o3,
    const float* x, int cols, uint8_t* q8_buf)
{
    quantize_for_type(x, q8_buf, t1, cols);
    dispatch_rows(w1, t1, q8_buf, cols, o1, r1, bpr1, 1);
    dispatch_rows(w2, t2, q8_buf, cols, o2, r2, bpr2, 1);
    dispatch_rows(w3, t3, q8_buf, cols, o3, r3, bpr3, 1);
}
