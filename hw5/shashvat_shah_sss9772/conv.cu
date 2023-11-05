#include <cuda_runtime.h>
#include <iostream>
#include "timer.h"

__global__ void conv2d_kernel(const double *inputTensor, const double *filterTensor, double *outputTensor, int H, int W, int C, int FH, int FW, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (x < W && y < H){
        double sum = 0;
        int filter_start_index = k * C * FH * FW;
        for (int c = 0; c < C; c++){
            int channel_start_index = c * FH * FW;
            int input_channel_start_index = c * (W + 2) * (H + 2);
            for (int j = 0; j < FH; j++){
                for (int i = 0; i < FW; i++){
                    sum += filterTensor[filter_start_index + channel_start_index + (FW - 1 - i) * FH + (FH - 1 - j)] * inputTensor[input_channel_start_index + (x + i) * (H + 2) + (y + j)];
                }
            }
        }
        outputTensor[k * W * H + x * H + y] = sum;
    }
}

double calcCheckSum(double *tensor, int K, int H, int W){
    double checksum = 0.0;
    for (int k = 0; k < K; ++k){
        for (int x = 0; x < W; ++x){
            for (int y = 0; y < H; ++y){
                checksum += tensor[k * W * H + x * H + y];
            }
        }
    }
    return checksum;
}

int main()
{
    // Set dimensions and allocate memory for the tensors
    const int H = 1024; // input tensor height
    const int W = 1024; // intput tensor width
    const int C = 3; // input tensor channels
    const int FH = 3; // filter tensor height 
    const int FW = 3; // filter tensor width
    const int K = 64; // output channel size
    const int P = 1; // padding

    double *inputTensor, *filterTensor, *outputTensor;
    cudaMallocManaged(&inputTensor, C * (W + 2 * P) * (H + 2 * P) * sizeof(double));
    cudaMallocManaged(&filterTensor, K * C * FH * FW * sizeof(double));
    cudaMallocManaged(&outputTensor, K * W * H * sizeof(double));

    // Initialize inputTensor and F tensors with the provided formulas
    for (int c = 0; c < C; c++){
        for (int x = 0; x < W + 2 * P; x++){
            for (int y = 0; y < H + 2 * P; y++){
                inputTensor[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y] = (x >= P && x < W + P && y >= P && y < H + P) ? c * (x - P + y - P) : 0;
            }
        }
    }

    for (int k = 0; k < K; k++){
        for (int c = 0; c < C; c++){
            for (int i = 0; i < FW; i++){
                for (int j = 0; j < FH; j++){
                    filterTensor[k * C * FH * FW + c * FH * FW + i * FH + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);
    // warmup

    conv2d_kernel<<<gridDim, blockDim>>>(inputTensor, filterTensor, outputTensor, H, W, C, FH, FW, K);

    // Launch the kernel

    initialize_timer();
    start_timer();
    conv2d_kernel<<<gridDim, blockDim>>>(inputTensor, filterTensor, outputTensor, H, W, C, FH, FW, K);
    stop_timer();
    cudaDeviceSynchronize();
    double time = elapsed_time();
    printf("Time: %lf (ms)\n", time*1000);
    // Report timing data.
    double checksum = calcCheckSum(outputTensor, K, H, W);
    // Print the checksum
    printf("C1_checksum = %f \n", checksum);
    // Deallocate memory
    cudaFree(inputTensor);
    cudaFree(filterTensor);
    cudaFree(outputTensor);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}