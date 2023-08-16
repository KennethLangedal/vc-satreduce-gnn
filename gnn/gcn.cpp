#include "gcn.h"

#include <immintrin.h>
#include <stdalign.h>
#include <stdlib.h>
#include <math.h>

#pragma GCC target("avx2")

// Specialized implementation for compile-time constant input dim

void gcn_layer_input(int N, int *V, int *E,
                     const float *__restrict__ W,
                     const float *__restrict__ B,
                     const float *__restrict__ x, float *__restrict__ y,
                     const float *__restrict__ scale)
{

    float _y[4][INPUT];

    for (int i = 0; i < N; i += 4)
    {
        // Compute message passing results
        for (int j = 0; j < 4; j++)
        {
            int u = i + j;
            if (u >= N)
                break;

            const float *_xu = x + u * INPUT; // Input for u
            for (int k = 0; k < INPUT; k++)
                _y[j][k] = scale[u] * _xu[k];

            // Neighbors
            for (int k = V[u]; k < V[u + 1]; k++)
            {
                int v = E[k];
                const float *_xv = x + v * INPUT; // Input for v
                for (int l = 0; l < INPUT; l++)
                    _y[j][l] += scale[v] * _xv[l];
            }

            for (int k = 0; k < INPUT; k++)
                _y[j][k] *= scale[u];
        }

        // Compute dense layer part
        for (int j = 0; j < DIM; j += 16)
        {

            __m256 c[4][2];
            for (int k = 0; k < 4; k++)
            {
                c[k][0] = _mm256_load_ps(&B[j]);
                c[k][1] = _mm256_load_ps(&B[j + 8]);
            }

            __m256 b0, b1, a;

            for (int k = 0; k < INPUT; k++)
            {
                b0 = _mm256_load_ps(&W[k * DIM + j]);
                b1 = _mm256_load_ps(&W[k * DIM + j + 8]);

                for (int l = 0; l < 4; l++)
                {
                    a = _mm256_broadcast_ss(&_y[l][k]);
                    c[l][0] = _mm256_fmadd_ps(a, b0, c[l][0]);
                    c[l][1] = _mm256_fmadd_ps(a, b1, c[l][1]);
                }
            }

            __m256 zero = _mm256_setzero_ps();
            for (int k = 0; k < 4; k++)
            {
                // ReLU before storing results
                c[k][0] = _mm256_max_ps(c[k][0], zero);
                c[k][1] = _mm256_max_ps(c[k][1], zero);

                _mm256_store_ps(&y[(i + k) * DIM + j], c[k][0]);
                _mm256_store_ps(&y[(i + k) * DIM + j + 8], c[k][1]);
            }
        }
    }
}

// Specialized implementation for compile-time constant hidden dim

void gcn_layer_hidden(int N, int *V, int *E,
                      const float *__restrict__ W,
                      const float *__restrict__ B,
                      const float *__restrict__ x, float *__restrict__ y,
                      const float *__restrict__ scale)
{

    alignas(32) float _y[4][DIM];
    const int dim_v = DIM / 8;

    for (int i = 0; i < N; i += 4)
    {
        // Compute message passing results
        for (int j = 0; j < 4; j++)
        {
            int u = i + j;
            if (u >= N)
                break;

            __m256 yv[dim_v];                           // Output for u
            __m256 su = _mm256_broadcast_ss(scale + u); // Scale for u
            const float *_xu = x + u * DIM;             // Input for u
            for (int k = 0; k < dim_v; k++)
            {
                __m256 xv = _mm256_load_ps(_xu + k * 8);
                yv[k] = _mm256_mul_ps(su, xv);
            }

            // Neighbors
            for (int k = V[u]; k < V[u + 1]; k++)
            {
                int v = E[k];
                __m256 sv = _mm256_broadcast_ss(scale + v); // Scale for v
                const float *_xv = x + v * DIM;             // Input for v
                for (int l = 0; l < dim_v; l++)
                {
                    __m256 xv = _mm256_load_ps(&_xv[l * 8]);
                    yv[l] = _mm256_fmadd_ps(sv, xv, yv[l]);
                }
            }

            for (int k = 0; k < dim_v; k++)
            {
                yv[k] = _mm256_mul_ps(yv[k], su);
                _mm256_store_ps(&_y[j][k * 8], yv[k]);
            }
        }

        // Compute dense layer part
        for (int j = 0; j < DIM; j += 16)
        {

            __m256 c[4][2];
            for (int k = 0; k < 4; k++)
            {
                c[k][0] = _mm256_load_ps(&B[j]);
                c[k][1] = _mm256_load_ps(&B[j + 8]);
            }

            __m256 b0, b1, a;

            for (int k = 0; k < DIM; k++)
            {
                b0 = _mm256_load_ps(&W[k * DIM + j]);
                b1 = _mm256_load_ps(&W[k * DIM + j + 8]);

                for (int l = 0; l < 4; l++)
                {
                    a = _mm256_broadcast_ss(&_y[l][k]);
                    c[l][0] = _mm256_fmadd_ps(a, b0, c[l][0]);
                    c[l][1] = _mm256_fmadd_ps(a, b1, c[l][1]);
                }
            }

            __m256 zero = _mm256_setzero_ps();
            for (int k = 0; k < 4; k++)
            {
                // ReLU before storing results
                c[k][0] = _mm256_max_ps(c[k][0], zero);
                c[k][1] = _mm256_max_ps(c[k][1], zero);

                _mm256_store_ps(&y[(i + k) * DIM + j], c[k][0]);
                _mm256_store_ps(&y[(i + k) * DIM + j + 8], c[k][1]);
            }
        }
    }
}

// Specialized implementation for compile-time constant output dim = 1

void gcn_layer_output(int N, int *V, int *E,
                      const float *__restrict__ W,
                      const float *__restrict__ B,
                      const float *__restrict__ x, float *__restrict__ y,
                      const float *__restrict__ scale)
{

    alignas(32) float _y[DIM];
    const int dim_v = DIM / 8;

    alignas(32) float _r[8];

    for (int i = 0; i < N; i += 8)
    {
        // Compute message passing results
        for (int j = 0; j < 8; j++)
        {
            int u = i + j;
            if (u >= N)
                break;

            __m256 yv[dim_v];                           // Output for u
            __m256 su = _mm256_broadcast_ss(scale + u); // Scale for u
            const float *_xu = x + u * DIM;             // Input for u
            for (int k = 0; k < dim_v; k++)
            {
                __m256 xv = _mm256_load_ps(_xu + k * 8);
                yv[k] = _mm256_mul_ps(su, xv);
            }

            // Neighbors
            for (int k = V[u]; k < V[u + 1]; k++)
            {
                int v = E[k];
                __m256 sv = _mm256_broadcast_ss(scale + v); // Scale for u
                const float *_xv = x + v * DIM;             // Input for v
                for (int l = 0; l < dim_v; l++)
                {
                    __m256 xv = _mm256_load_ps(&_xv[l * 8]);
                    yv[l] = _mm256_fmadd_ps(sv, xv, yv[l]);
                }
            }

            for (int k = 0; k < dim_v; k++)
            {
                yv[k] = _mm256_mul_ps(yv[k], su);
                _mm256_store_ps(&_y[k * 8], yv[k]);
            }

            // Compute dense layer part
            __m256 c = _mm256_setzero_ps();
            for (int k = 0; k < DIM; k += 8)
            {
                __m256 a = _mm256_load_ps(&_y[k]);
                __m256 b = _mm256_load_ps(&W[k]);
                c = _mm256_fmadd_ps(a, b, c);
            }
            __m128 l = _mm256_extractf128_ps(c, 0);
            __m128 h = _mm256_extractf128_ps(c, 1);
            l = _mm_add_ps(l, h);
            l = _mm_hadd_ps(l, l);
            l = _mm_hadd_ps(l, l);

            _r[j] = _mm_cvtss_f32(l) + B[0];
        }

        __m256 r = _mm256_load_ps(_r);
        _mm256_store_ps(y + i, r);
    }
}

gcn gcn_init(int hidden_layers)
{
    gcn m = {.hidden_layers = hidden_layers,
             .N = 0,
             .params = NULL,
             .W = NULL,
             .B = NULL};

    m.N = DIM * 2 + hidden_layers * (DIM * DIM + DIM) + DIM * 2;
    m.params = (float *)aligned_alloc(32, sizeof(float) * m.N);
    m.W = (float **)malloc(sizeof(float *) * (hidden_layers + 2));
    m.B = (float **)malloc(sizeof(float *) * (hidden_layers + 2));

    float *p = m.params;

    m.W[0] = p;
    p += DIM;
    m.B[0] = p;
    p += DIM;

    for (int i = 0; i < hidden_layers; i++)
    {
        m.W[i + 1] = p;
        p += DIM * DIM;
        m.B[i + 1] = p;
        p += DIM;
    }

    m.W[hidden_layers + 1] = p;
    p += DIM;
    m.B[hidden_layers + 1] = p;

    for (int i = 0; i < m.N; i++)
        m.params[i] = 0.0f;

    return m;
}

gcn_data gcn_data_init(int N, int *V, int *E)
{
    gcn_data md = {.N = N, .V = V, .E = E};
    md.scale = (float *)malloc(sizeof(float) * N);
    md.x = (float *)aligned_alloc(32, sizeof(float) * DIM * ((N + 7) & ~7));
    md.y = (float *)aligned_alloc(32, sizeof(float) * DIM * ((N + 7) & ~7));
    return md;
}

void gcn_store(FILE *f, gcn m)
{
    fprintf(f, "%d %d %d\n", m.hidden_layers, INPUT, DIM);
    for (int i = 0; i < m.N; i++)
        fprintf(f, "%.10f ", m.params[i]);

    fprintf(f, "\n");
}

gcn gcn_parse(FILE *f)
{
    int layers, in, dim;
    int t = fscanf(f, "%d %d %d\n", &layers, &in, &dim);
    if (t != 3 || in != INPUT || dim != DIM)
    {
        fprintf(stderr, "Model incompatible with build\n");
        exit(1);
    }
    gcn m = gcn_init(layers);
    for (int i = 0; i < m.N; i++)
    {
        t = fscanf(f, "%f ", &m.params[i]);
        if (t != 1)
        {
            fprintf(stderr, "Wrong number of params\n");
            exit(1);
        }
    }
    return m;
}

void gcn_free(gcn m)
{
    free(m.params);
    free(m.W);
    free(m.B);
}

void gcn_data_free(gcn_data md)
{
    free(md.scale);
    free(md.x);
    free(md.y);
}

void swap(float **a, float **b)
{
    float *t = *a;
    *a = *b;
    *b = t;
}

void gcn_eval(gcn m, gcn_data md)
{
    int N = md.N;
    int *V = md.V, *E = md.E;
    float *x = md.x, *y = md.y, *scale = md.scale;

    int l = m.hidden_layers;

    for (int i = 0; i < N; i++)
        scale[i] = 1.0f / sqrtf((V[i + 1] - V[i]) + 1.0f);

    for (int i = 0; i < N; i++)
        x[i] = 1.0f;

    gcn_layer_input(N, V, E, m.W[0], m.B[0], x, y, scale);
    swap(&x, &y);

    for (int i = 0; i < l; i++)
    {
        gcn_layer_hidden(N, V, E, m.W[i + 1], m.B[i + 1], x, y, scale);
        swap(&x, &y);
    }

    gcn_layer_output(N, V, E, m.W[l + 1], m.B[l + 1], x, y, scale);
    swap(&x, &y);

    if (!(l & 1))
    {
        for (int i = 0; i < N; i++)
            y[i] = x[i];
    }
}