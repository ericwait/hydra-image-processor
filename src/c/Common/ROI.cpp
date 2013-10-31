#include "ROI.h"
#include <string>

void getROI(const HostPixelType* imageIn, HostPixelType* imageOut, Vec<unsigned int> inDims, Vec<unsigned int> startIdx,
			Vec<unsigned int> sizes)
{
	if (startIdx==Vec<unsigned int>(0,0,0) && sizes==inDims)
	{
		memcpy(imageOut,imageIn,sizeof(HostPixelType)*inDims.product());
		return;
	}

	Vec<unsigned int> curIdxIn(startIdx);
	Vec<unsigned int> curIdxOut(0,0,0);
	for (curIdxIn.z=startIdx.z, curIdxOut.z=0; curIdxIn.z<startIdx.z+sizes.z && curIdxIn.z<inDims.z; ++curIdxIn.z, ++curIdxOut.x)
	{
		for (curIdxIn.y=startIdx.y, curIdxOut.y=0; curIdxIn.y<startIdx.y+sizes.y && curIdxIn.y<inDims.y; ++curIdxIn.y, ++curIdxOut.y)
		{
			for (curIdxIn.x=startIdx.x, curIdxOut.x=0; curIdxIn.x<startIdx.x+sizes.x && curIdxIn.x<inDims.x; ++curIdxIn.x, ++curIdxOut.x)
			{
				imageOut[sizes.linearAddressAt(curIdxOut)] = imageIn[inDims.linearAddressAt(curIdxIn)];
			}
		}
	}
}

void replaceROI(const HostPixelType* imageIn, HostPixelType* imageOut, Vec<unsigned int> outDims, Vec<unsigned int> startIdx,
				Vec<unsigned int> sizes)
{
	if (startIdx==Vec<unsigned int>(0,0,0) && sizes==outDims)
	{
		memcpy(imageOut,imageIn,sizeof(HostPixelType)*outDims.product());
		return;
	}

	Vec<unsigned int> curIdxIn(0,0,0);
	Vec<unsigned int> curIdxOut(startIdx);
	for (curIdxIn.z=0, curIdxOut.z=startIdx.z; curIdxIn.z<sizes.z && curIdxOut.z<outDims.z; ++curIdxIn.z, ++curIdxOut.z)
	{
		for (curIdxIn.y=0, curIdxOut.y=startIdx.y; curIdxIn.y<sizes.y && curIdxOut.y<outDims.y; ++curIdxIn.y, ++curIdxOut.y)
		{
			for (curIdxIn.x=0, curIdxOut.x=startIdx.x; curIdxIn.x<sizes.x && curIdxOut.x<outDims.x; ++curIdxIn.x, ++curIdxOut.x)
			{
				imageOut[outDims.linearAddressAt(curIdxOut)] = imageIn[sizes.linearAddressAt(curIdxIn)];
			}
		}
	}
}