#pragma once
#include "Vec.h"

typedef unsigned char HostPixelType;

void clear();
void addConstant(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, double additive);
void addImageWith(const HostPixelType* image1, const HostPixelType* image2, HostPixelType* imageOut, Vec<unsigned int> imageDims,
				  double factor);
void applyPolyTransformation(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, double a, double b,
							 double c, HostPixelType minValue, HostPixelType maxValue);
void calculateMinMax(const HostPixelType* image, Vec<unsigned int> imageDims, double& minValue, double& maxValue);
void contrastEnhancement(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas,
						 Vec<unsigned int> medianNeighborhood);
void gaussianFilter(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas);
size_t getGlobalMemoryAvailable(const HostPixelType* image1, const HostPixelType* image2, HostPixelType* imageOut,
								Vec<unsigned int> imageDims, double factor);
void mask(const HostPixelType* image1, const HostPixelType* image2, HostPixelType* imageOut, Vec<unsigned int> imageDims,
		  double theshold=1);
void maxFilter(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
			   double* kernel=NULL);
void maximumIntensityProjection(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims);
void meanFilter(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood);
void medianFilter(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood);
void minFilter(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
			   double* kernel=NULL);
void morphClosure(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
				  double* kernel=NULL);
void morphOpening(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
				  double* kernel=NULL);
void multiplyImage(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, double factor);
void multiplyImageWith(const HostPixelType* image1, const HostPixelType* image2, HostPixelType* imageOut,
					   Vec<unsigned int> imageDims);
double normalizedCovariance(const HostPixelType* image1, const HostPixelType* image2, Vec<unsigned int> imageDims);
void otsuThresholdFilter(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, double alpha);
HostPixelType otsuThesholdValue(const HostPixelType* image, Vec<unsigned int> imageDims);
void imagePow(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, int p);
double sumArray(const HostPixelType* image, Vec<unsigned int> imageDims);
HostPixelType* reduceImage(const HostPixelType* image, Vec<unsigned int>& imageDims, Vec<double> reductions);
unsigned int* retrieveHistogram(const HostPixelType* image, Vec<unsigned int>& imageDims, int& returnSize);
double* retrieveNormalizedHistogram(const HostPixelType* image, Vec<unsigned int>& imageDims, int& returnSize);
void thresholdFilter(const HostPixelType* image, HostPixelType* imageOut, Vec<unsigned int> imageDims, double threshold);
void unmix(const HostPixelType* image, const HostPixelType* image2, HostPixelType* imageOut, Vec<unsigned int> imageDims,
		   Vec<unsigned int> neighborhood);
