#pragma once
#include "Vec.h"

typedef unsigned char MexImagePixelType;

void addConstant(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double additive);
void addImageWith(const MexImagePixelType* image1, MexImagePixelType* imageOut, const MexImagePixelType* image2, Vec<unsigned int> imageDims, double factor);
void applyPolyTransformation(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double a, double b, double c, MexImagePixelType minValue, MexImagePixelType maxValue);
void calculateMinMax(const MexImagePixelType* image, Vec<unsigned int> imageDims, MexImagePixelType& minValue, MexImagePixelType& maxValue);
void contrastEnhancement(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas, Vec<unsigned int> medianNeighborhood);
void gaussianFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas);
void maxFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood);
void maximumIntensityProjection(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims);
void meanFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood);
void medianFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood);
void minFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood);
void multiplyImage(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double factor);
void multiplyImageWith(const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut, Vec<unsigned int> imageDims);
void imagePow(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, int p);
double sumArray(const MexImagePixelType* image, Vec<unsigned int> imageDims);
MexImagePixelType* reduceImage(const MexImagePixelType* image, Vec<unsigned int>& imageDims, Vec<double> reductions);
void thresholdFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double threshold);
void unmix(const MexImagePixelType* image, const MexImagePixelType* image2, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood);