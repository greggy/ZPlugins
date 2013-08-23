#include <stdio.h>
#include <string.h>

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <Exceptions.h>

#include <npp.h>
#include "utils.h"


void boxfilter1_transform( Npp8u *data, int width, int height ){
    size_t size = width * height * 4;

    // declare a host image object for an 8-bit RGBA image
    npp::ImageCPU_8u_C4 oHostSrc(width, height);

    //printf("Pitch: %d\n", oHostSrc.pitch());

    Npp8u *nDstData = oHostSrc.data();
    memcpy(nDstData, data, size * sizeof(Npp8u));
    //printf("Size: %d\n", oHostSrc.data());

    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C4 oDeviceSrc(oHostSrc);

    // create struct with box-filter mask size
    NppiSize oMaskSize = {3, 3};

    // Allocate memory for pKernel
    Npp32s hostKernel[9] = {0, 0, -2, 0, 2, 0, 1, 0, 0}; // reverse order
    Npp32s *pKernel;

    checkCudaErrors( cudaMalloc((void**)&pKernel, oMaskSize.width * oMaskSize.height * sizeof(Npp32s)) );
    checkCudaErrors( cudaMemcpy(pKernel, hostKernel, oMaskSize.width * oMaskSize.height * sizeof(Npp32s),
                                cudaMemcpyHostToDevice) );

    Npp32s nDivisor = 1;

    // create struct with ROI size given the current mask
    NppiSize oSizeROI = {oDeviceSrc.width() - oMaskSize.width + 1, oDeviceSrc.height() - oMaskSize.height + 1};
    // allocate device image of appropriatedly reduced size
    npp::ImageNPP_8u_C4 oDeviceDst(oSizeROI.width, oSizeROI.height);
    // set anchor point inside the mask
    NppiPoint oAnchor = {2, 2};

    // run box filter
    NppStatus eStatusNPP;
    eStatusNPP = nppiFilter_8u_C4R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                   oDeviceDst.data(), oDeviceDst.pitch(),
                                   oSizeROI, pKernel, oMaskSize, oAnchor, nDivisor);
    //printf("NppiFilter error status %d\n", eStatusNPP);
    NPP_DEBUG_ASSERT(NPP_NO_ERROR == eStatusNPP);

    // declare a host image for the result
    npp::ImageCPU_8u_C4 oHostDst(width, height);
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    memcpy(data, oHostDst.data(), size * sizeof(Npp8u));

    return;
}


void boxfilter2_transform( Npp8u *data, int width, int height ){
    size_t size = width * height * 4;

    Npp8u *oData;
    for (int i = 0; i < width * height; i++){
        int pixelId = i * 4;
        oData[i] = data[pixelId + 2]; // get only red channel
    }

    // declare a host image object for an 8-bit RGBA image
//    npp::ImageCPU_8u_C1 oHostSrc(width, height);

//    Npp8u *nDstData = oHostSrc.data();
//    memcpy(nDstData, oData, width * height * sizeof(Npp8u));

//    // declare a device image and copy construct from the host image,
//    // i.e. upload host to device
//    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

//    // create struct with box-filter mask size
//    NppiSize oMaskSize = {5, 5};
//    // create struct with ROI size given the current mask
//    NppiSize oSizeROI = {width - oMaskSize.width + 1, height - oMaskSize.height + 1};
//    // allocate device image of appropriatedly reduced size
//    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
//    // set anchor point inside the mask to (0, 0)
//    NppiPoint oAnchor = {0, 0};
//    // run box filter
//    NppStatus eStatusNPP;
//    eStatusNPP = nppiFilterBox_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
//                                      oDeviceDst.data(), oDeviceDst.pitch(),
//                                      oSizeROI, oMaskSize, oAnchor);
//    NPP_DEBUG_ASSERT(NPP_NO_ERROR == eStatusNPP);
//    // declare a host image for the result
//    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
//    // and copy the device result data into it
//    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

//    Npp8u *oDataBack = oHostDst.data();
//    for (int i = 0; i < width * height; i++){
//        int pixelId = i * 4;
//        data[pixelId] = 0; // blue
//        data[pixelId + 1] = 0; // green
//        data[pixelId + 2] = oDataBack[i]; // set only red channel
//    }
//    //checkCudaErrors( cudaMemcpy( data, oDeviceDst.data(), size, cudaMemcpyDeviceToHost ) );

    return;
}
