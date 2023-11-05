// vecadd.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// Add two Vectors A and B in C on GPU using
// a kernel defined according to vecAddKernel.h

// DO NOT MODIFY FOR THE ASSIGNMENT

// Includes
#include <stdio.h>
#include "timer.h"

// Defines


// Variables for host and device vectors.
float *h_A;
float *h_B;
float *h_C;
float *d_A;
float *d_B;
float *d_C;

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

__global__ void AddVectorKernel(const float *A, const float *B, float *C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (i; i < N; i += stride){
        C[i] = A[i] + B[i];
    }
}

// Host code performs setup and calls the kernel.
int main(int argc, char **argv)
{
    int sizeOfArray; // number of values per thread
    int N;               // Vector size
    int GridWidth = 1;
    int BlockWidth = 256;
    // Parse arguments.
    if (argc != 2){
        printf("Usage: %s sizeOfArray\n", argv[0]);
        exit(0);
    }
    else{
        sscanf(argv[1], "%d", &sizeOfArray);
    }

    // Determine the number of threads .
    // N is the total number of values to be in a vector
    N = sizeOfArray * 1000000;
    printf("Total vector size: %d\n", N);
    // size_t is the total number of bytes for a vector.
    size_t size = N * sizeof(float);

    // Tell CUDA how big to make the grid and thread blocks.
    // Since this is a vector addition problem,
    // grid and thread block are both one-dimensional.
    dim3 dimGrid(GridWidth);
    dim3 dimBlock(BlockWidth);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Allocate vectors in device memory.
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Initialize host vectors h_A and h_B
    int i;
    for (i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Copy host vectors h_A and h_B to device vectores d_A and d_B
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Warm up
    AddVectorKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
        Cleanup(false);
    cudaDeviceSynchronize();

    // Initialize timer
    initialize_timer();
    start_timer();

    // Invoke kernel
    AddVectorKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        Cleanup(false);
    cudaDeviceSynchronize();

    // Compute elapsed time
    stop_timer();
    double time = elapsed_time();

    // Compute floating point operations per second.
    int nFlops = N;
    double nFlopsPerSec = nFlops / time;
    double nGFlopsPerSec = nFlopsPerSec * 1e-9;

    // Compute transfer rates.
    int nBytes = 3 * 4 * N; // 2N words in, 1N word out
    double nBytesPerSec = nBytes / time;
    double nGBytesPerSec = nBytesPerSec * 1e-9;

    // Report timing data.
    printf("Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n",
           time, nGFlopsPerSec, nGBytesPerSec);
    cudaDeviceSynchronize();
    // Copy result from device memory to host memory
    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        Cleanup(false);
    
    float errorvals = 0;
    // Verify & report result
    for (i = 0; i < N; ++i)
    {
        float val = h_C[i];
        if (fabs(val - N) > 1e-5)
            errorvals += float(val - N);
    }
    printf("Test %s \n", abs(errorvals) < 1e-5 ? "PASSED" : "FAILED");
    printf("Error: %f", errorvals);

    // Clean up and exit.
    Cleanup(true);
}

void Cleanup(bool noError)
{ // simplified version from CUDA SDK
    cudaError_t error;

    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);

    error = cudaDeviceReset();

    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");

    fflush(stdout);
    fflush(stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
