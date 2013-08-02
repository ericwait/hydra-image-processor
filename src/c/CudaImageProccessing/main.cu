#include "CudaImageBuffer.h"

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

}