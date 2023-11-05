// matmultKernel00.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// Multiplies two matrices using CUDA: A x B = C

// Copy this file and modify the MatMultKernel device function for
// any of your experiments. 

#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue[2][2];
  memset(Cvalue, 0, sizeof(Cvalue));

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Each thread copies just one element of shared_A and one element of shared_B
#pragma unroll
    for (int i = 0; i < 2; i++){
      int tr_new = thread_row * 2 + i;
      for (int j = 0; j < 2; j++){
          int tc_new = thread_col * 2 + j;
          shared_A[tr_new][tc_new] = Asub[tr_new * A.stride + tc_new];
          shared_B[tr_new][tc_new] = Bsub[tr_new * B.stride + tc_new];
      }
    }
    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    for (int i = 0; i < 2; i++){
      int tr_new = thread_row * 2 + i;
      for (int j = 0; j < 2; j++){
          int tc_new = thread_col * 2 + j;
          for (int e = 0; e < FOOTPRINT_SIZE; ++e){
              Cvalue[i][j] += shared_A[tr_new][e] * shared_B[e][tc_new];
          }
      }
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
// #pragma unroll
//     for(int p=0;p<4;p++){
//         Csub[thread_row * C.stride + thread_col] += cvalue[p];
//     }
#pragma unroll
    for (int i = 0; i < 2; i++){
        int tr_new = thread_row * 2 + i;
        for (int j = 0; j < 2; j++){
            int tc_new = thread_col * 2 + j;
            Csub[tr_new * C.stride + tc_new] = Cvalue[i][j];
        }
    }
}

