// vecadd.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// Add two Vectors A and B in C on GPU using
// a kernel defined according to vecAddKernel.h

// DO NOT MODIFY FOR THE ASSIGNMENT

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

__global__ void AddVectorKernel(const float* A, const float* B, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

void checkCUDAError(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void cleanup(float* d_A, float* d_B, float* d_C) {
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaDeviceReset();
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s sizeOfArray\n", argv[0]);
    return EXIT_FAILURE;
  }

  int sizeOfArray;
  if (sscanf(argv[1], "%d", &sizeOfArray) != 1) {
    printf("Invalid sizeOfArray argument: %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  int N = sizeOfArray * 1000000;
  size_t size = N * sizeof(float);

  float *d_A, *d_B, *d_C;
  cudaMallocManaged(&d_A, size);
  cudaMallocManaged(&d_B, size);
  cudaMallocManaged(&d_C, size);

  for (int i = 0; i < N; ++i) {
    d_A[i] = (float)i;
    d_B[i] = (float)(N - i);
  }

  dim3 dimGrid(1);
  dim3 dimBlock(1);

  AddVectorKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
  checkCUDAError("AddVectorKernel");

  cudaDeviceSynchronize();

  initialize_timer();
  start_timer();

  AddVectorKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
  checkCUDAError("AddVectorKernel");

  cudaDeviceSynchronize();

  stop_timer();
  double time = elapsed_time();

  int nFlops = N;
  double nFlopsPerSec = nFlops / time;
  double nGFlopsPerSec = nFlopsPerSec * 1e-9;

  int nBytes = 3 * 4 * N;
  double nBytesPerSec = nBytes / time;
  double nGBytesPerSec = nBytesPerSec * 1e-9;

  float errorvals = 0;
  for (int i = 0; i < N; ++i) {
    float val = d_C[i];
    if (fabs(val - N) > 1e-5)
      errorvals += float(val - N);
  }

  printf("Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", time, nGFlopsPerSec, nGBytesPerSec);
  printf("Test %s \n", fabs(errorvals) < 1e-5 ? "PASSED" : "FAILED");
  printf("Error: %f\n", errorvals);

  cleanup(d_A, d_B, d_C);

  return EXIT_SUCCESS;
}
