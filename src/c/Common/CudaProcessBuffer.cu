#include "CudaUtilities.cuh"
#include "Vec.h"
#include "CudaProcessBuffer.cuh"
#include "CudaDeviceImages.cuh"
#include "CHelpers.h"
#include "CudaImageReduction.cuh"
#include "CudaMeanFilter.cuh"

CudaProcessBuffer::CudaProcessBuffer(int device/*=0*/)
{
	defaults();
	this->device = device;
	deviceSetup();
}

CudaProcessBuffer::~CudaProcessBuffer()
{
	defaults();
}

void CudaProcessBuffer::deviceSetup()
{
	HANDLE_ERROR(cudaSetDevice(device));
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
}

void CudaProcessBuffer::defaults()
{
	device = 0;
	orgImageDims = Vec<size_t>(0,0,0);
	maxDeviceDims = Vec<size_t>(0,0,0);
}

//////////////////////////////////////////////////////////////////////////
//Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////


double CudaProcessBuffer::normalizedCovariance(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims)
{
// 	double im1Mean = sumArray(imageIn1,dims.product()) / dims.product();
// 	double im2Mean = sumArray(imageIn2,dims.product()) / dims.product();
// 
// 	DevicePixelType* im1Sub = addConstant(imageIn1,dims,-1.0*im1Mean);
// 	DevicePixelType* im2Sub = addConstant(imageIn2,dims,-1.0*im2Mean);
// 
// 	DevicePixelType* im1P = imagePow(im1Sub,dims,2.0);
// 	DevicePixelType* im2P = imagePow(im2Sub,dims,2.0);
// 
// 	double sigma1 = sqrt(sumArray(im1P,dims.product())/dims.product());
// 	double sigma2 = sqrt(sumArray(im2P,dims.product())/dims.product());
// 
// 	DevicePixelType* imMul = multiplyImageWith(im1Sub,im2Sub,dims,1.0);
// 	double numarator = sumArray(imMul,dims.product());
// 
// 	double coVar = numarator/(dims.product()*sigma1*sigma2);
// 
// 	delete[] im1Sub;
// 	delete[] im2Sub;
// 	delete[] im1P;
// 	delete[] im2P;
// 	delete[] imMul;
// 
// 	return coVar;

	return 0.0;
}

