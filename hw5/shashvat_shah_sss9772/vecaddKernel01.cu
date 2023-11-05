// vecAddKernel00.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// This Kernel adds two Vectors A and B in C on GPU
// without using coalesced memory access.

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int threadIndex  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(threadIndex;threadIndex < stride*N; threadIndex+=stride)
        C[threadIndex] = A[threadIndex] + B[threadIndex];   
}
