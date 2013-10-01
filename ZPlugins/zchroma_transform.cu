#include <stdio.h>
#include <sys/param.h>
#include <math.h>

#include <helper_functions.h>
#include <helper_cuda.h>

typedef unsigned char guint8;



__global__ void mask_kernel(
        guint8 *data,
        guint8 *a_data,
        int width,
        int height
    )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int pix = (y * width + x) * 4; // main pixel
    int pixNum = (y * width + x);

    float b = data[pix];
    float g = data[pix + 1];
    float r = data[pix + 2];
    //float a = data[pix + 3];

    float rd = 100 - r * 100 / g;
    if (r < g && b < g && rd > 35)
    //if (g > r + delta && g > b + delta)
    {
        a_data[pixNum] = 0;
    } else {
        a_data[pixNum] = 255;
    }
}


__global__ void gaussian_kernel_rgb(
        guint8 *data,
        guint8 *o_data,
        int width,
        int height,
        int filterWidth,
        float *filter
    )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ( x >= width || y >= height )
        return;

    int pixelId = (y * width + x) * 4;

    int center = filterWidth / 2;
    float blur_b, blur_g, blur_r = 0;

    for(int i = -center; i <= center; i++)
    {
        for(int j = -center; j <= center; j++)
        {
          int jY = min(max(y + j, 0), static_cast<int>(height - 1));
          int iX = min(max(x + i, 0), static_cast<int>(width - 1));
          int npixel = (jY * width + iX) * 4;
          float d_filter = filter[(j + center) * filterWidth + (i + center)];
          blur_b += d_filter * (float)data[npixel];
          blur_g += d_filter * (float)data[npixel + 1];
          blur_r += d_filter * (float)data[npixel + 2];
        }
    }

    o_data[pixelId] = blur_b;
    o_data[pixelId + 1] = blur_g;
    o_data[pixelId + 2] = blur_r;
    o_data[pixelId + 3] = 255;
}


__global__ void gaussian_kernel_alpha(
        guint8 *data,
        guint8 *o_data,
        int width,
        int height,
        int filterWidth,
        float *filter
    )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ( x >= width || y >= height )
        return;

    int pixelId = (y * width + x);
    int center = filterWidth / 2;
    float blur = 0;

    for(int i = -center; i <= center; i++)
    {
        for(int j = -center; j <= center; j++)
        {
          int jY = min(max(y + j, 0), static_cast<int>(height - 1));
          int iX = min(max(x + i, 0), static_cast<int>(width - 1));
          int npixel = (jY * width + iX);
          float d_filter = filter[(j + center) * filterWidth + (i + center)];
          blur += d_filter * (float)data[npixel];
        }
    }

    o_data[pixelId] = blur;
}


__global__ void chroma_transform(
        guint8 *alpha_data,
        guint8 *data,
        guint8 *o_data,
        int width,
        int height
    )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int pix = (y * width + x); // main pixel

    o_data[pix * 4] = data[pix * 4];
    o_data[pix * 4 + 1] = data[pix * 4 + 1];
    o_data[pix * 4 + 2] = data[pix * 4 + 2];
    if (alpha_data[pix])
    {
        o_data[pix * 4 + 3] = 255;
    } else {
        o_data[pix * 4 + 3] = alpha_data[pix];
    }
}


void zchroma_transform( guint8 *data, int width, int height, int filterWidth, float *filter )
{
    guint8 *o_data, *d_data, *alpha_data, *d_alpha_data, *h_alpha_data;
    size_t size = width * height * 4;
    float *d_filter;

    h_alpha_data = (guint8*)malloc( sizeof(guint8)*width*height );
    checkCudaErrors( cudaMalloc( (void**)&d_data, size ) );
    checkCudaErrors( cudaMalloc( (void**)&o_data, size ) );
    checkCudaErrors( cudaMalloc( (void**)&alpha_data, width*height ) );
    checkCudaErrors( cudaMalloc( (void**)&d_alpha_data, width*height ) );
    checkCudaErrors( cudaMemcpy( d_data, data, size, cudaMemcpyHostToDevice ) );

    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3(width / threads.x + 1, height / threads.y + 1, 1);

    checkCudaErrors( cudaMalloc( (void**)&d_filter, sizeof(float)*filterWidth*filterWidth ) );
    checkCudaErrors( cudaMemcpy( d_filter, filter, sizeof(float)*filterWidth*filterWidth, cudaMemcpyHostToDevice ) );

    // gaussian blur kernel rgb
    gaussian_kernel_rgb<<< blocks, threads >>>( d_data, o_data, width, height, filterWidth, d_filter );

    // mask kernel
    mask_kernel<<< blocks, threads >>>( o_data, alpha_data, width, height );

    // gaussian blur kernel alpha
    gaussian_kernel_alpha<<< blocks, threads >>>( alpha_data, d_alpha_data, width, height, filterWidth, d_filter );

    // main kernel
    chroma_transform<<< blocks, threads >>>( d_alpha_data, d_data, o_data, width, height );

    checkCudaErrors( cudaMemcpy( data, o_data, size, cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy( h_alpha_data, d_alpha_data, width*height, cudaMemcpyDeviceToHost ) );

//    for (int i = 0; i < width*height; i++){
//        printf("Pixel %d: value %d\n", i, h_alpha_data[i]);
//    }
//    for (int i = 0; i < width*height; i++){
//        int b = data[i * 4];
//        int g = data[i * 4 + 1];
//        int r = data[i * 4 + 2];
//        int a = data[i * 4 + 3];
//        printf("Pixel %d: {b: %d, g: %d, r: %d, a: %d}\n", i, b, g, r, a);
//    }

    cudaFree( d_data );
    cudaFree( o_data );
    cudaFree( alpha_data );
    cudaFree( d_alpha_data );
    cudaFree( d_filter );
}
