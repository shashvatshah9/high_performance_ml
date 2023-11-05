
#ifndef __CONVTILEDKERNEL__
#define __CONVTILEDKERNEL__

#define BLOCK_SIZE 8
typedef struct{
    int width;
    int height;
    int depth;
    double *elements;
} Matrix3d;

typedef struct{
    int width;
    int height;
    int depth;
    int layer;
    double *elements;
} Matrix4d;

__global__ void convolution(Matrix3d inputMatrix, Matrix4d filterMatrix, Matrix3d outputMatrix);
#endif