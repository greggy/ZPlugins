#include <stdio.h>
#include <sys/param.h>
#include "utils.h"

typedef unsigned char guint8;

#define BLOCK_DIM 16


__global__ void zcartoon_kernel(
        guint8 *data,
        int width,
        int height,
        int top,
        int m_mask_radius,
        float m_threshold,
        float m_ramp,
        float m_size
        )
    {

    //__shared__ float temp[BLOCK_DIM][BLOCK_DIM];

    int x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int y = ((blockIdx.y * blockDim.y) + threadIdx.y);

    int m_pixelPos = (y * width + x) * 4; // main pixel


//    if (x < width && y < height)
//        temp[x][y] = data[m_pixelPos];
//    __syncthreads();


    // get neighbour pixels
    int i = 0;
    int sumR = 0, sumB = 0, sumG = 0;
    for(int iX = x-top; i < m_mask_radius; ++i, ++iX){

      int j = 0;
      for(int iY = y-top; j < m_mask_radius; ++j, ++iY){

        uint n_pixelPos = (iY * width + iX) * 4; // neighbour pixel
        sumR += data[n_pixelPos + 2];
        sumB += data[n_pixelPos + 0];
        sumG += data[n_pixelPos + 1];
      }
    }

    sumR /= m_size;
    sumB /= m_size;
    sumG /= m_size;

    int red = data[m_pixelPos + 2],
           blue = data[m_pixelPos + 0],
           green = data[m_pixelPos + 1];

    int koeffR = red / sumR,
           koeffB = blue / sumB,
           koeffG = green / sumG;

    if(koeffR < m_threshold)
        red *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffR)))/m_ramp);

    if(koeffB < m_threshold)
        blue *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffB)))/m_ramp);

    if(koeffG < m_threshold)
        green *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffG)))/m_ramp);

    data[m_pixelPos + 2] = red;
    data[m_pixelPos + 0] = blue;
    data[m_pixelPos + 1] = green;

}

void zcartoon_transform( guint8 *data, int width, int height ){
    guint8 *d_data;
    int m_mask_radius = 7;
    float m_threshold = 1.0;
    float m_ramp = 0.1;
    size_t size = width * height * 4;

    float m_size = m_mask_radius * m_mask_radius;
    int top = m_mask_radius / 2;

    checkCudaErrors( cudaMalloc( (void**)&d_data, size ) );
    checkCudaErrors( cudaMemcpy( d_data, data, size, cudaMemcpyHostToDevice ) );

    dim3 threads = dim3(32, 32);
    dim3 blocks = dim3(width / threads.x, height / threads.y);
    //printf("Threads x: %d, y: %d; blocks x: %d, y: %d\n", threads.x, threads.y, blocks.x, blocks.y);

    // execute kernel
    zcartoon_kernel<<< blocks, threads >>>( d_data, width, height, top, m_mask_radius, m_threshold, m_ramp, m_size );

    checkCudaErrors( cudaMemcpy( data, d_data, size, cudaMemcpyDeviceToHost ) );

    cudaFree( d_data );
}
