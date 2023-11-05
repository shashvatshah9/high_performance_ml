#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include "convTiledKernel.h"

int main(int argc, char *argv[]){
    // Set dimensions and allocate memory for the tensors
    const int H = 1024; // input tensor height
    const int W = 1024; // intput tensor width
    const int C = 3; // input tensor channels
    const int FH = 3; // filter tensor height 
    const int FW = 3; // filter tensor width
    const int K = 64; // output channel size

    // Host input vectors
    Matrix3d h_input;
    Matrix4d h_filter;
    Matrix3d h_output;

    // Device input vectors
    Matrix3d d_input;
    Matrix4d d_filter;
    Matrix3d d_output;

    // Allocate memory for each vector on host
    int H_P = H + 2; // padded height
    int W_P = W + 2; // padded width

    h_input.width = W_P, h_input.height = H_P, h_input.depth = C;
    size_t input_size = C * H_P * W_P * sizeof(double);
    h_input.elements = (double *)malloc(input_size);

    h_filter.width = FW, h_filter.height = FH, h_filter.depth = C, h_filter.layer = K;
    size_t filter_size = K * C * FH * FW * sizeof(double);
    h_filter.elements = (double *)malloc(filter_size);

    h_output.width = W, h_output.height = H, h_output.depth = K;
    size_t output_size = K * W * H * sizeof(double);
    h_output.elements = (double *)malloc(output_size);

    // Allocate memory for each vector on GPU
    d_input.width = W_P, d_input.height = H_P, d_input.depth = C;
    cudaMalloc((void **)&d_input.elements, input_size);
    d_filter.width = FW, d_filter.height = FH, d_filter.depth = C, d_filter.layer = K;
    cudaMalloc((void **)&d_filter.elements, filter_size);
    d_output.width = W, d_output.height = H, d_output.depth = K;
    cudaMalloc((void **)&d_output.elements, output_size);

    // Initialize input on host
    // input[c,x,y]=h_input[c*H_P*W_P+x*W_P+y], dimension C,H,W

    for (int c = 0; c < C; c++){
        int channel_offset = c * H_P * W_P;
        for (int x = 1; x <= H; x++){
            int row_offset = x * W_P;
            for (int y = 1; y <= W; y++){
                h_input.elements[channel_offset + row_offset + y] = c * (x - 1 + y - 1);
            }
        }
        for (int x = 0; x < H_P; x++){
            h_input.elements[channel_offset + x * W_P] = 0;
            h_input.elements[channel_offset + x * W_P + (W_P - 1)] = 0;
        }
        for (int y = 0; y < W_P; y++){
            h_input.elements[channel_offset + y] = 0;
            h_input.elements[channel_offset + (H_P - 1) * W_P + y] = 0;
        }
    }

    for (int k = 0; k < K; k++){
        for (int c = 0; c < C; c++){
            for (int i = 0; i < FH; i++){
                int channel_offset = c * FH * FW;
                int filter_offset = k * C * FH * FW + channel_offset + i * FW;
                for (int j = 0; j < FW; j++){
                    h_filter.elements[filter_offset + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Copy host vectors to device
    cudaMemcpy(d_input.elements, h_input.elements, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter.elements, h_filter.elements, filter_size, cudaMemcpyHostToDevice);

    // Execute the kernel
    dim3 dimGrid(W / BLOCK_SIZE, H / BLOCK_SIZE, K / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    
    // warmup    
    convolution<<<dimGrid, dimBlock>>>(d_input, d_filter, d_output);

    initialize_timer();
    start_timer();
    convolution<<<dimGrid, dimBlock>>>(d_input, d_filter, d_output);
    stop_timer();
    double time = elapsed_time();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));


    // Copy array back to host
    cudaMemcpy(h_output.elements, d_output.elements, output_size, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within error
    printf("Time: %lf (ms)\n", time*1000);
    double checksum = 0.0;
    for (int i = 0; i < K * W * H; i++)
    {
        checksum += h_output.elements[i];
        // printf("%d %f\n",i,h_output.elements[i]);
    }
    printf("C2_checksum = %f \n", checksum);

    // Release device memory
    cudaFree(d_input.elements);
    cudaFree(d_filter.elements);
    cudaFree(d_output.elements);

    // Release host memory
    free(h_input.elements);
    free(h_filter.elements);
    free(h_output.elements);

    return 0;
}
