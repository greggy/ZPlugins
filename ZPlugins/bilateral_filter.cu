#include <stdio.h>

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

typedef unsigned char guint8;


__constant__ float cGaussian[64];   //gaussian array in device side
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;

guint8 *dImage  = NULL;   //original image
guint8 *dTemp  = NULL;   //temp data
size_t pitch;


//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt2(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

//column pass using coalesced global memory reads
__global__ void
bilateral_transform(guint8 *od, int w, int h,
                    float e_d,  int r)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float4 t = {0.f, 0.f, 0.f, 0.f};
    float4 center = tex2D(rgbaTex, x, y);

    for (int i = -r; i <= r; i++)
    {
        for (int j = -r; j <= r; j++)
        {
            float4 curPix = tex2D(rgbaTex, x + j, y + i);
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                     euclideanLen(curPix, center, e_d);             //range factor

            t += factor * curPix;
            sum += factor;
        }
    }

    int pixId = (y * w + x) * 4;

    //od[pixId] = rgbaFloatToInt2(pixel);
    uint tmp = rgbaFloatToInt2(t/sum);
    memcpy(&od[pixId], &tmp, sizeof(tmp));
}


void updateGaussian(float delta, int radius)
{
    float  fGaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));
    }

    checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1)));
}


void bilateral_transform( guint8 *data, int width, int height ){
    guint8 *d_data;
    float gaussian_delta = 4;
    float e_d = 0.1f;
    int radius = 5;
    int iterations = 5;
    size_t size = width * height * 4;

    updateGaussian(gaussian_delta, radius);

    // copy image data to array
    checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(guint8) * width * 4, height));
    checkCudaErrors(cudaMallocPitch(&dTemp,  &pitch, sizeof(guint8) * width * 4, height));
    checkCudaErrors(cudaMalloc(&d_data, size));
    checkCudaErrors(cudaMemcpy2D(dImage, pitch, data, sizeof(guint8) * width * 4,
                                 sizeof(guint8) * width * 4, height, cudaMemcpyHostToDevice));
    //printf("Pitch: %d\n", pitch);

    // bind array to texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, width, height, pitch));

    for (int i=0; i<iterations; i++)
    {
        dim3 threads = dim3(32, 32, 1);
        dim3 blocks = dim3(width / threads.x + 1, height / threads.y + 1);
        bilateral_transform<<<blocks, threads>>>(d_data, width, height, e_d, radius);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, d_data, sizeof(guint8) * width * 4,
                                         sizeof(guint8) * width * 4, height, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
        }
    }

    checkCudaErrors(cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dImage));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaUnbindTexture(rgbaTex));
}

/*
// 1D example without iterations
void bilateral2_transform( guint8 *data, int width, int height ){
    guint8 *d_data;
    float gaussian_delta = 4;
    float e_d = 0.1f;
    int radius = 5;
    size_t size = width * height * 4;

    updateGaussian(gaussian_delta, radius);

    // copy image data to array
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&dImage, &desc, width, height));
    checkCudaErrors(cudaMalloc(&d_data, size));
    checkCudaErrors(cudaMemcpyToArray(dImage, 0, 0, data, size, cudaMemcpyHostToDevice));

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(rgbaTex, dImage));

    dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
    dim3 blockSize(16, 16);
    bilateral_transform<<< gridSize, blockSize >>>(d_data, width, height, e_d, radius);

    checkCudaErrors(cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFreeArray(dImage));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaUnbindTexture(rgbaTex));

}
*/
