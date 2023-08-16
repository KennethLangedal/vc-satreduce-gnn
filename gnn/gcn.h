#pragma once

#include <stdio.h>

#define INPUT 1
#define DIM 32

typedef struct
{
    int hidden_layers, N;
    float *params, **W, **B;
} gcn;

typedef struct
{
    int N;
    int *V, *E;
    float *x, *y, *scale;
} gcn_data;

gcn gcn_init(int hidden_layers);

gcn_data gcn_data_init(int N, int *V, int *E);

void gcn_store(FILE *f, gcn m);

gcn gcn_parse(FILE *f);

void gcn_free(gcn m);

void gcn_data_free(gcn_data md);

void gcn_eval(gcn m, gcn_data md);
