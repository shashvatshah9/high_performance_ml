#include <assert.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "timer.h"

#define CHECK_CUDNN(ans) {checkCudnnAssert((ans), __FILE__, __LINE__); }
#define CHECK_CUDA(ans) {checkCudaAssert((ans), __FILE__, __LINE__); }

inline void checkCudnnAssert(cudnnStatus_t code, const char *file, int line, bool abort=true){
    if (code != CUDNN_STATUS_SUCCESS){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        if(abort) assert(0);
    }
}

inline void checkCudaAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){                                                        
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) assert(0);
    }   
}

int main(){
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Initialize input tensor descriptor
    cudnnTensorDescriptor_t inputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, 3, 1026, 1026));

    // Initialize filter descriptor
    cudnnFilterDescriptor_t filterDesc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 64, 3, 3, 3));

    // Initialize convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    // Allocate GPU memory
    double *inputTensor, *filterTensor, *outputTensor;
    const size_t imageSize = 1 * 3 * 1026 * 1026 * sizeof(double);
    const size_t filterSize = 64 * 3 * 3 * 3 * sizeof(double);
    const size_t outputSize = 1 * 64 * 1024 * 1024 * sizeof(double);
    cudaMalloc(&inputTensor, imageSize);
    cudaMalloc(&filterTensor, filterSize);
    cudaMalloc(&outputTensor, outputSize);

    // Initialize input tensor (inputTensor) with padding
    std::vector<double> hostInputTensor(imageSize / sizeof(double), 0);
    for (int c = 0; c < 3; ++c){
        for (int x = 0; x < 1024; ++x){
            for (int y = 0; y < 1024; ++y){
                hostInputTensor[((c * (1026)) + (x + 1)) * 1026 + (y + 1)] = c * (x + y);
            }
        }
    }

    // Initialize filter tensor (F)
    std::vector<double> hostFilterTensor(filterSize / sizeof(double));
    for (int k = 0; k < 64; ++k){
        for (int c = 0; c < 3; ++c){
            for (int i = 0; i < 3; ++i){
                for (int j = 0; j < 3; ++j){
                    hostFilterTensor[((k * 3) + c) * 9 + i * 3 + j] = (c + k) * (i + j);
                }
            }
        }
    }
    cudaMemcpy(inputTensor, hostInputTensor.data(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(filterTensor, hostFilterTensor.data(), filterSize, cudaMemcpyHostToDevice);

    // Find fastest convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t algoPerf[1000];
    int algoCount = 1;
    const int algoC = 1;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, inputDesc, algoC, &algoCount, algoPerf));
    cudnnConvolutionFwdAlgo_t algo = algoPerf[0].algo;

    size_t workspaceSize;
    // Allocate workspace

    // Perform convolution
    cudnnTensorDescriptor_t outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, out_n, out_c, out_h, out_w));

    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));
    void *d_workspace;
    cudaMalloc(&d_workspace, workspaceSize);
    const double alpha = 1.0, beta = 0.0;

    // warmup
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, inputTensor, filterDesc, filterTensor, convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, outputTensor));
    // Perform convolution

    initialize_timer();
    start_timer();
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, inputTensor, filterDesc, filterTensor, convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, outputTensor);
    stop_timer();
   
    double time = elapsed_time();
    printf("Time: %lf (ms)\n",time*1000);
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
    size_t output_elements = out_n * out_c * out_h * out_w;

    thrust::device_ptr<double> d_output_ptr(outputTensor);
    double checksum = thrust::reduce(d_output_ptr, d_output_ptr + output_elements, 0.0, thrust::plus<double>());

    // Print the checksum
    printf("C3_Checksum: %f\n", checksum);
    // Clean up cuDNN descriptors and handle
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroy(cudnn);

    // Deallocate GPU memory
    cudaFree(inputTensor);
    cudaFree(filterTensor);
    cudaFree(outputTensor);
    cudaFree(d_workspace);

    return 0;
}
