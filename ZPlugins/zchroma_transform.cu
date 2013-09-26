#include <stdio.h>
#include <sys/param.h>

#include <helper_functions.h>
#include <helper_cuda.h>

typedef unsigned char guint8;



__global__ void zchroma_kernel(
        guint8 *data,
        guint8 *o_data,
        int width,
        int height
    )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int pix = (y * width + x) * 4; // main pixel

    float b = data[pix];
    float g = data[pix + 1];
    float r = data[pix + 2];
    float a = data[pix + 3];

//   if ((r + b)/(r + g + b) < 0.05f)
    if (g > r && g > b)
    {
        o_data[pix] = 0;
        o_data[pix + 1] = 0;
        o_data[pix + 2] = 0;
        o_data[pix + 3] = 255 - (g - max(r, b));
    } else {
        o_data[pix] = b;
        o_data[pix + 1] = g;
        o_data[pix + 2] = r;
        o_data[pix + 3] = a;
    }

//    o_data[pix] = b;
//    o_data[pix + 1] = 0;
//    o_data[pix + 2] = r;
//    o_data[pix + 3] = a;
}


void zchroma_transform( guint8 *data, int width, int height )
{
    guint8 *o_data, *d_data;
    size_t size = width * height * 4;

    checkCudaErrors( cudaMalloc( (void**)&d_data, sizeof(guint8)*size ) );
    checkCudaErrors( cudaMalloc( (void**)&o_data, sizeof(guint8)*size ) );
    checkCudaErrors( cudaMemcpy( d_data, data, sizeof(guint8)*size, cudaMemcpyHostToDevice ) );

//    for (int i = 0; i < width*height; i++){
//        int b = data[i * 4];
//        int g = data[i * 4 + 1];
//        int r = data[i * 4 + 2];
//        int a = data[i * 4 + 3];
//        printf("Pixel %d: {b: %d, g: %d, r: %d, a: %d}\n", i, b, g, r, a);
//    }

    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3(width / threads.x + 1, height / threads.y + 1, 1);

    // execute kernel
    zchroma_kernel<<< blocks, threads >>>( d_data, o_data, width, height );

    checkCudaErrors( cudaMemcpy( data, o_data, sizeof(guint8)*size, cudaMemcpyDeviceToHost ) );

    cudaFree( d_data );
    cudaFree( o_data );

}
