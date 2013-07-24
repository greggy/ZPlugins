#include <stdio.h>
#include <sys/param.h>

#include "utils.h"
#include "vector_types.h"

#include <helper_math.h>


typedef unsigned char guint8;

#define BLOCK_DIM 21


// kernel with global memory
__global__ void zcartoon_kernel(
        guint8 *data,
        guint8 *o_data,
        int width,
        int height,
        int top,
        int m_mask_radius,
        float m_threshold,
        float m_ramp,
        float m_size
        )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int pixelPos = (y * width + x) * 4; // main pixel

    // get neighbour pixels
    int i = 0;
    int sumR = 0, sumB = 0, sumG = 0;
    for(int iX = x-top; i < m_mask_radius; i++, iX++){

      int j = 0;
      for(int iY = y-top; j < m_mask_radius; j++, iY++){

        int n_pixelPos = (iY * width + iX) * 4; // neighbour pixel
        sumR += data[n_pixelPos + 2];
        sumB += data[n_pixelPos + 0];
        sumG += data[n_pixelPos + 1];
      }
    }

    float blue = data[pixelPos],
          green = data[pixelPos + 1],
          red = data[pixelPos + 2];

    float koeffR = red / (sumR / m_size),
           koeffB = blue / (sumB / m_size),
           koeffG = green / (sumG / m_size);

    if(koeffR < m_threshold)
        red *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffR)))/m_ramp);

    if(koeffB < m_threshold)
        blue *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffB)))/m_ramp);

    if(koeffG < m_threshold)
        green *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffG)))/m_ramp);

    o_data[pixelPos + 0] = blue;
    o_data[pixelPos + 1] = green;
    o_data[pixelPos + 2] = red;
    o_data[pixelPos + 3] = data[pixelPos + 3];

}


// kernel with shared memory
__global__ void zcartoon2_kernel(
        guint8 *data,
        guint8 *o_data,
        int width,
        int height,
        int top,
        int m_mask_radius,
        float m_threshold,
        float m_ramp,
        float m_size
        )
{

    __shared__ float4 temp[BLOCK_DIM+6][BLOCK_DIM+6];

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(x < 3 || x >= width-3 || y < 3 || y >= height-3)
        return;

    unsigned int shY = threadIdx.y + 3;
    unsigned int shX = threadIdx.x + 3;

    int pixelPos = (y * width + x) * 4; // main pixel

    float4 pixel = {
        o_data[pixelPos],
        o_data[pixelPos + 1],
        o_data[pixelPos + 2],
        o_data[pixelPos + 3]
    };

    temp[shY][shX] = pixel;

    __syncthreads();

    // get neighbour pixels
    int sumR = 0, sumB = 0, sumG = 0;
    for(int i = -top; i <= top; i++){
      for(int j = -top; j <= top; j++){
        float4 npixel = temp[shY+i][shX+j];
        sumB += npixel.x; // blue
        sumG += npixel.y; // green
        sumR += npixel.z; // red
      }
    }

    float blue = o_data[pixelPos],
          green = o_data[pixelPos + 1],
          red = o_data[pixelPos + 2];

    float koeffR = red / (sumR / m_size),
           koeffB = blue / (sumB / m_size),
           koeffG = green / (sumG / m_size);

    if(koeffR < m_threshold)
        red *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffR)))/m_ramp);

    if(koeffB < m_threshold)
        blue *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffB)))/m_ramp);

    if(koeffG < m_threshold)
        green *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffG)))/m_ramp);

    data[pixelPos + 0] = blue;
    data[pixelPos + 1] = green;
    data[pixelPos + 2] = red;
    data[pixelPos + 3] = o_data[pixelPos + 3];

}


texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
cudaArray *d_data;

// convert floating point rgba color to 32-bit integer
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

// kernel with texture memory
__global__ void zcartoon3_kernel(
        guint8 *o_data,
        int width,
        int height,
        int top,
        int m_mask_radius,
        float m_threshold,
        float m_ramp,
        float m_size
        )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float4 blurPixel = make_float4(0.0f);
    float koeffCore = float(m_mask_radius);

    // get neighbour pixels
    int i = 0;
    for(int iX = x-top; i < m_mask_radius; i++,iX++)
    {
        int j = 0;
        for(int jY = y-top; j < m_mask_radius; j++,jY++)
        {
            float4 nPix = tex2D(rgbaTex, iX, jY);
            blurPixel += nPix;
        }
    }

    // get main pixel
    float4 pixel = tex2D(rgbaTex, x, y);
    blurPixel /= (koeffCore*koeffCore);

    float4 koeff = pixel / blurPixel;

    // blue
    if(koeff.x < m_threshold)
        pixel.x *= ((m_ramp - MIN(m_ramp,(m_threshold - koeff.x)))/m_ramp);

    // green
    if(koeff.y < m_threshold)
        pixel.y *= ((m_ramp - MIN(m_ramp,(m_threshold - koeff.y)))/m_ramp);

    // red
    if(koeff.z < m_threshold)
        pixel.z *= ((m_ramp - MIN(m_ramp,(m_threshold - koeff.z)))/m_ramp);


    int pixelPos = (y * width + x) * 4;
    uint tmp = rgbaFloatToInt(pixel);
    memcpy(&o_data[pixelPos], &tmp, sizeof(tmp));

}


void zcartoon_transform( guint8 *data, int width, int height ){
    guint8 *o_data;
    int m_mask_radius = 7;
    float m_threshold = 1.0;
    float m_ramp = 0.1;
    size_t size = width * height * 4;

    float m_size = m_mask_radius * m_mask_radius;
    int top = m_mask_radius / 2;

    // for global and shared memory
//    checkCudaErrors( cudaMalloc( (void**)&d_data, size ) );
//    checkCudaErrors( cudaMalloc( (void**)&o_data, size ) );
//    //checkCudaErrors( cudaMemcpy( d_data, data, size, cudaMemcpyHostToDevice ) );
//    checkCudaErrors( cudaMemcpy( o_data, data, size, cudaMemcpyHostToDevice ) );

//    dim3 threads = dim3(21, 21, 1);
//    dim3 blocks = dim3(width / threads.x, height / threads.y, 1);
//    //printf("Threads x: %d, y: %d; blocks x: %d, y: %d\n", threads.x, threads.y, blocks.x, blocks.y);

//    // execute kernel
//    zcartoon2_kernel<<< blocks, threads >>>( d_data, o_data, width, height, top, m_mask_radius, m_threshold, m_ramp, m_size );

//    checkCudaErrors( cudaMemcpy( data, d_data, size, cudaMemcpyDeviceToHost ) );

//    cudaFree( d_data );
//    cudaFree( o_data );

    // for texture memory
    // copy image data to array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    checkCudaErrors( cudaMallocArray ( &d_data, &channelDesc, width, height ));
    checkCudaErrors( cudaMalloc( &o_data, size ));
    checkCudaErrors( cudaMemcpyToArray( d_data, 0, 0, data, size, cudaMemcpyHostToDevice));

    // normalization false
    rgbaTex.addressMode[0] = cudaAddressModeWrap; // режим Wrap
    rgbaTex.addressMode[1] = cudaAddressModeWrap;
    rgbaTex.filterMode =  cudaFilterModePoint;  // ближайшее значение
    rgbaTex.normalized = false; // не использовать нормализованную адресацию

    checkCudaErrors( cudaBindTextureToArray(rgbaTex, d_data) );

    dim3 threads = dim3(32, 32, 1);
    dim3 blocks = dim3(width / threads.x, height / threads.y, 1);

    zcartoon3_kernel<<< blocks, threads >>>(o_data, width, height, top, m_mask_radius, m_threshold, m_ramp, m_size);

    checkCudaErrors(cudaMemcpy(data, o_data, size, cudaMemcpyDeviceToHost));

//    for (int i = 0; i < width * height; i++)
//    {
//        printf("Blue: %d, green: %d, red: %d, alpha: %d\n", data[i], data[i+1], data[i+2], data[i+3]);
//    }

    checkCudaErrors(cudaFreeArray(d_data));
    checkCudaErrors(cudaFree(o_data));
    checkCudaErrors(cudaUnbindTexture(rgbaTex));

}
