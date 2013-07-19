#include <stdio.h>
#include <sys/param.h>
#include "utils.h"

typedef unsigned char guint8;

#define BLOCK_DIM 16


__global__ void simple_kernel( guint8 *data, float *o_data, int width, int height )
{
    __shared__ float temp[BLOCK_DIM][BLOCK_DIM];

    int x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int y = ((blockIdx.y * blockDim.y) + threadIdx.y);

    int pixelPos = (y * width + x) * 4;

    if (x < width && y < height){
        temp[x][y] = data[pixelPos];
    }
    __syncthreads();

    x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    y = ((blockIdx.y * blockDim.y) + threadIdx.y);

    if (x < width && y < height){
        //int pixelPos = temp[x][y];
        o_data[pixelPos] = temp[x][y];
    }
}

void simple_transform( guint8 *data, int width, int height ){
    guint8 *d_data;
    float *o_data;
    size_t size = width * height * 4;

    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    checkCudaErrors( cudaMalloc( (void**)&d_data, size ) );
    checkCudaErrors( cudaMalloc( (void**)&o_data, size ) );
    checkCudaErrors( cudaMemcpy( d_data, data, size, cudaMemcpyHostToDevice ) );

    //dim3 threads = dim3(8, 8);
    //dim3 blocks = dim3(width / threads.x, height / threads.y);
    dim3 threads = dim3(width / BLOCK_DIM, height / BLOCK_DIM);
    dim3 blocks = dim3(BLOCK_DIM, BLOCK_DIM);

    // execute kernel
    simple_kernel<<< blocks, threads >>>( d_data, o_data, width, height );

    checkCudaErrors( cudaMemcpy( data, o_data, size, cudaMemcpyDeviceToHost ) );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
//    printf("Frame was proccessed during: %f\n", elapsedTime);

    cudaFree( d_data );
    cudaFree( o_data );
}


// test kernel
__global__ void add( int a, int b, int *c){
           *c = a + b;
}

int test( int len ){
    int c;
    int *dev_c;

    checkCudaErrors( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

    add<<<1, 1>>>( 2, 7, dev_c );

    checkCudaErrors( cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ) );

    printf( "2 + 7 = %d and %d\n", c, len );
    cudaFree( dev_c );

    return 0;
}
