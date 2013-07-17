#include <stdio.h>
#include <sys/param.h>
#include "utils.h"

typedef unsigned char guint8;


__global__ void simple_kernel( guint8 *data, int width )
    {
    uint x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    uint y = ((blockIdx.y * blockDim.y) + threadIdx.y);

    uint pixelPos = (y * width + x) * 4;
    data[pixelPos + 2] = 0;
}

void simple_transform( guint8 *data, int i_width, int i_height ){
    guint8 *d_data;
    size_t size = i_width * i_height * 4;

    //printf ("Image length %d", len);


    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    checkCudaErrors( cudaMalloc( (void**)&d_data, size ) );
    checkCudaErrors( cudaMemcpy( d_data, data, size, cudaMemcpyHostToDevice ) );

    dim3 threads = dim3(8, 8);
    dim3 blocks = dim3(i_width / threads.x, i_height / threads.y);

    // execute kernel
    simple_kernel<<< blocks, threads >>>( d_data, i_width );

    checkCudaErrors( cudaMemcpy( data, d_data, size, cudaMemcpyDeviceToHost ) );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
//    printf("Frame was proccessed during: %f\n", elapsedTime);

    cudaFree( d_data );
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
