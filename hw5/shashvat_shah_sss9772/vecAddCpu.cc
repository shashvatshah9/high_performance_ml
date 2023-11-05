#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include "timer.h"

void AddElements(float *h_A, float *h_B, float *h_C, int size) {
    for (int i = 0; i < size; ++i) {
        h_C[i] = h_A[i] + h_B[i];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s sizeOfArray\nTotal vector size is 10e6 * this value.\n", argv[0]);
        exit(0);
    }

    int k = atoi(argv[1]);
    int size = 1000000 * k;
    float *h_A = (float*) malloc(size * sizeof(float));
    float *h_B = (float*) malloc(size * sizeof(float));
    float *h_C = (float*) malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        h_A[i] = (float) i;
        h_B[i] = (float) (size - i);
    }

    initialize_timer();
    start_timer();

    AddElements(h_A, h_B, h_C, size);

    stop_timer();

    double time = elapsed_time();
    double nFlopsPerSec = size / time;
    double nGFlopsPerSec = nFlopsPerSec * 1e-9;
    int nBytes = 3 * 4 * size; // 2N words in, 1N word out
    double nBytesPerSec = nBytes / time;
    double nGBytesPerSec = nBytesPerSec * 1e-9;

    printf("Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", time, nGFlopsPerSec, nGBytesPerSec);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
