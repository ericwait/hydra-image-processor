#include "CudaImageBuffer.cuh"

#define XSIZE 100
#define YSIZE 100
#define ZSIZE 100

int main(int argc, char* argv[])
{
	unsigned char ucharImage[XSIZE*YSIZE*ZSIZE];
	unsigned int uintImage[XSIZE*YSIZE*ZSIZE];
	float floatImage[XSIZE*YSIZE*ZSIZE];
	double doubleImage[XSIZE*YSIZE*ZSIZE];

	CudaImageBuffer<unsigned char> ucharBuffer(XSIZE,YSIZE,ZSIZE);
	CudaImageBuffer<float> fBuffer(Vec<unsigned int>(XSIZE,YSIZE,ZSIZE));

	ucharBuffer.addConstant((int)1);
	ucharBuffer.addImageWith(&ucharBuffer,1.0);
	ucharBuffer.applyPolyTransformation(1,1,1,0,255);
	unsigned char mn,mx;
	ucharBuffer.calculateMinMax(mn,mx);
	ucharBuffer.createHistogram();
	ucharBuffer.gaussianFilter(Vec<double>(1.0,1.0,1.0));
	ucharBuffer.maxFilter(Vec<int>(3,3,3));
	ucharBuffer.maximumIntensityProjection();
	ucharBuffer.meanFilter(Vec<int>(3,3,3));
	ucharBuffer.medianFilter(Vec<int>(3,3,3));
	ucharBuffer.minFilter(Vec<int>(3,3,3));
	ucharBuffer.multiplyImage(3.0,0,255);
	ucharBuffer.multiplyImageWith(&ucharBuffer);
	ucharBuffer.normalizeHistogram();
	ucharBuffer.pow(2.0);
	double sum;
	ucharBuffer.sumArray(sum);
	ucharBuffer.reduceImage(Vec<double>(3.0,3.0,3.0));
	ucharBuffer.thresholdFilter(25);
}