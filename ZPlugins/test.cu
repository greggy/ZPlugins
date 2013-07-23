#include <stdio.h>
#include <sys/param.h>
#include "utils.h"

typedef unsigned char guint8;

#define BLOCK_DIM 16


// I = 0.2126 * R + 0.7152 * G + 0.0722 * B
__global__ void gray_kernel( guint8 *data, unsigned char* const gray, int width, int height )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int pixelPos = (y * width + x) * 4;

    float b = data[pixelPos];
    float g = data[pixelPos + 1];
    float r = data[pixelPos + 2];
    //int a = data[pixelPos + 3];

    int grayPos = (y * width + x);
    gray[grayPos] = (float)0.2126 * r + (float)0.7152 * g + (float)0.0722 * b;
}


__global__ void simple_kernel( guint8 *data, guint8 *o_data, int width, int height )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int pixelPos = (y * width + x) * 4;
    data[pixelPos + 2] = data[pixelPos + 2] / 2;

//    __shared__ float temp[BLOCK_DIM][BLOCK_DIM];

//    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

//    if (x < width && y < height){
//        int pixelPos = (y * width + x) * 4;

//        temp[threadIdx.y][threadIdx.x] = data[pixelPos + 2];
//    }
//    __syncthreads();

//    x = (blockIdx.x * blockDim.x) + threadIdx.x;
//    y = (blockIdx.y * blockDim.y) + threadIdx.y;

//    if (x < width && y < height){
//        int pixelPos = (y * width + x) * 4;
//        o_data[pixelPos + 2] = temp[threadIdx.y][threadIdx.x];
//    }
}

void simple_transform( guint8 *data, int width, int height ){
    guint8 *d_data;
    unsigned char *gray;
    size_t size = width * height * 4;
    size_t g_size = width * height;

    checkCudaErrors( cudaMalloc( (void**)&d_data, size ) );
    checkCudaErrors( cudaMalloc( (void**)&gray, g_size ) );
    checkCudaErrors( cudaMemcpy( d_data, data, size, cudaMemcpyHostToDevice ) );

    //dim3 threads = dim3(16, 16);
    //dim3 blocks = dim3(width / threads.x, height / threads.y);
    dim3 threads = dim3(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks = dim3(width / BLOCK_DIM, height / BLOCK_DIM);

    // execute kernel
    gray_kernel<<< blocks, threads >>>( d_data, gray, width, height );

    checkCudaErrors( cudaMemcpy( data, gray, size, cudaMemcpyDeviceToHost ) );

    cudaFree( d_data );
    cudaFree( gray );
}


// test kernel
__global__ void add( int a, int b, int *c){
           *c = a + b;
}

int test( int len ){
    int c;
    int *dev_c;

//    cudaEvent_t start, stop;
//    cudaEventCreate( &start );
//    cudaEventCreate( &stop );
//    cudaEventRecord( start, 0 );

    checkCudaErrors( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

    add<<<1, 1>>>( 2, 7, dev_c );

    checkCudaErrors( cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ) );

//    cudaEventRecord( stop, 0 );
//    cudaEventSynchronize( stop );
//    float   elapsedTime;
//    cudaEventElapsedTime( &elapsedTime, start, stop );
//    printf("Frame was proccessed during: %f\n", elapsedTime);

    printf( "2 + 7 = %d and %d\n", c, len );
    cudaFree( dev_c );

    return 0;
}
