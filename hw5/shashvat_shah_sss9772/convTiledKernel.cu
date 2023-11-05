
#include "convTiledKernel.h"


__global__ void convolution(Matrix3d inputMatrix, Matrix4d filterMatrix, Matrix3d outputMatrix){
    const int C = inputMatrix.depth;  // input channels 
    const int FH = filterMatrix.height; // filter height
    const int FW = filterMatrix.width;  // filter width
    const int W_P = inputMatrix.width;  // input image width
    const int H_P = inputMatrix.height; // input image height
    const int H = outputMatrix.height;  // output height
    const int W = outputMatrix.width;   // output width

    int block_dep = blockIdx.z;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    Matrix3d output_sub{
        .width = BLOCK_SIZE,
        .height = BLOCK_SIZE,
        .depth = BLOCK_SIZE,
        .elements = &outputMatrix.elements[block_dep * BLOCK_SIZE * W * H + block_row * BLOCK_SIZE * W + block_col * BLOCK_SIZE]};

    double output_value = 0;
    int thread_row = threadIdx.y;
    int thread_dep = threadIdx.z;
    int thread_col = threadIdx.x;

    Matrix3d input_sub{
        .width = BLOCK_SIZE + 2,
        .height = BLOCK_SIZE + 2,
        .depth = C,
        .elements = &inputMatrix.elements[block_row * BLOCK_SIZE * W_P + block_col * BLOCK_SIZE]};

    Matrix4d filter_sub{
        .width = FW,
        .height = FH,
        .depth = C,
        .layer = BLOCK_SIZE,
        .elements = &filterMatrix.elements[block_dep * C * FH * FW * BLOCK_SIZE]};

    __shared__ double input_s[3][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ double filter_s[BLOCK_SIZE][3][3][3];

#pragma unroll
    for (int c = 0; c < C; ++c){
        int index_start = c * H_P * W_P;
        for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                input_s[c][thread_row + i][thread_col + j] = input_sub.elements[index_start + (thread_row + i) * W_P + thread_col + j];
            }
        }
    }

    int thread_start = thread_dep * C * FH * FW;
#pragma unroll
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            for (int c = 0; c < C; ++c){
                filter_s[thread_dep][c][i][j] = filter_sub.elements[thread_start + c * FH * FW + i * FW + j];
            }
        }
    }

    __syncthreads();
#pragma unroll
    for (int c = 0; c < C; c++){
        for (int j = 0; j < FH; j++){
            for (int i = 0; i < FW; i++){
                output_value += filter_s[thread_dep][c][FW - 1 - i][FH - 1 - j] * input_s[c][thread_row + i][thread_col + j];
            }
        }
    }

    __syncthreads();

    output_sub.elements[thread_dep * H * W + thread_row * W + thread_col] = output_value;
}
