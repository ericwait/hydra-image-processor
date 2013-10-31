#include "Process.h"

void getROI(const HostPixelType* imageIn, HostPixelType* imageOut, Vec<unsigned int> inDims, Vec<unsigned int> startIdx, 
			Vec<unsigned int> sizes);
void replaceROI(const HostPixelType* imageIn, HostPixelType* imageOut, Vec<unsigned int> outDims, Vec<unsigned int> startIdx,
				Vec<unsigned int> sizes);
