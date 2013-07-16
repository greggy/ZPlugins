#include <stdio.h>
#include <sys/param.h>
#include "utils.h"

typedef unsigned char guint8;


__global__ void transform(
        guint8 *data,
        guint8 *o_data,
        int width,
        int top,
        int m_mask_radius,
        double m_threshold,
        double m_ramp,
        double m_size
        )
    {
    uint x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    uint y = ((blockIdx.y * blockDim.y) + threadIdx.y);

    uint m_pixelPos = (y * width + x) * 4; // main pixel

    // get neighbour pixels
    int i = 0;
    double sumR = 0, sumB = 0, sumG = 0;
    for(int iX = x-top; i < m_mask_radius; ++i, ++iX){

      int j = 0;
      for(int iY = y-top; j < m_mask_radius; ++j, ++iY){

        uint n_pixelPos = (iY * width + iX) * 4; // neighbour pixel
        sumR += o_data[n_pixelPos + 2];
        sumB += o_data[n_pixelPos + 0];
        sumG += o_data[n_pixelPos + 1];
      }
    }

    sumR /= m_size;
    sumB /= m_size;
    sumG /= m_size;

    double red = o_data[m_pixelPos + 2],
           blue = o_data[m_pixelPos + 0],
           green = o_data[m_pixelPos + 1];

    double koeffR = red / sumR,
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

void zcartoon_transform( guint8 *data, guint8 *o_data, int i_width, int i_height ){
    guint8 *d_data;
    guint8 *do_data;
    int m_mask_radius = 7;
    double m_threshold = 1.0;
    double m_ramp = 0.1;
    size_t size = i_width * i_height * 4;

    //printf ("Image length %d", len);

    //int center = m_mask_radius / 2 + 1,
            //width = i_width - center,
            //height = i_height - center,
    int top = m_mask_radius / 2;


    checkCudaErrors( cudaMalloc( (void**)&d_data, size ) );
    checkCudaErrors( cudaMalloc( (void**)&do_data, size ) );
    checkCudaErrors( cudaMemcpy( d_data, data, size, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( do_data, data, size, cudaMemcpyHostToDevice ) );

    dim3 threads = dim3(8, 8);
    dim3 blocks = dim3(i_width / threads.x, i_height / threads.y);

    double m_size = m_mask_radius * m_mask_radius;

    // execute kernel
    transform<<< blocks, threads >>>( d_data, do_data, i_width, top, m_mask_radius, m_threshold, m_ramp, m_size );

    checkCudaErrors( cudaMemcpy( data, d_data, size, cudaMemcpyDeviceToHost ) );

//    for (int i = 0; i < width * height; i++){
//        int b = data[i + 0];
//        int g = data[i + 1];
//        int r = data[i + 2];
//        if (b > 0 || g > 0 || r > 0)
//            printf ("%d, %d, %d\n", b, g, r);
//    }

    cudaFree( d_data );
    cudaFree( do_data );
    free( o_data );
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
