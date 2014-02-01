//  #pragma once
//  #include "Vec.h"
//  #include <stdio.h>
//  #include "ImageContainer.h"
// 
// void clearAll();
// void addConstant(const ImageContainer* image,  ImageContainer* imageOut, double additive, int deviceNum=0);
// void addImageWith(const ImageContainer* image1, const ImageContainer* image2, ImageContainer* imageOut, double factor);
// void applyPolyTransformation(const ImageContainer* image, ImageContainer* imageOut, double a, double b, double c, HostPixelType minValue,
// 							 HostPixelType maxValue);
// void calculateMinMax(const ImageContainer* image, double& minValue, double& maxValue);
// void contrastEnhancement(const ImageContainer* image, ImageContainer* imageOut, Vec<float> sigmas, Vec<size_t> medianNeighborhood);
// void gaussianFilter(const ImageContainer* image, ImageContainer* imageOut, Vec<float> sigmas);
// size_t getGlobalMemoryAvailable(const ImageContainer* image1, const ImageContainer* image2, ImageContainer* imageOut, double factor);
// void mask(const ImageContainer* image1, const ImageContainer* image2, ImageContainer* imageOut, double theshold=1);
// void maxFilter(const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood, double* kernel=NULL);
// void maximumIntensityProjection(const ImageContainer* image, ImageContainer* imageOut);
// void meanFilter(const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood);
// void medianFilter(const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood);
// void minFilter(const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood, double* kernel=NULL);
// void morphClosure(const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood, double* kernel=NULL);
// void morphOpening(const ImageContainer* image, ImageContainer* imageOut, Vec<size_t> neighborhood, double* kernel=NULL);
// void multiplyImage(const ImageContainer* image, ImageContainer* imageOut, double factor);
// void multiplyImageWith(const ImageContainer* image1, const ImageContainer* image2, ImageContainer* imageOut);
// double normalizedCovariance(const ImageContainer* image1, const ImageContainer* image2);
// void otsuThresholdFilter(const ImageContainer* image, ImageContainer* imageOut, double alpha);
// HostPixelType otsuThesholdValue(const ImageContainer* image);
// void imagePow(const ImageContainer* image, ImageContainer* imageOut, int p);
// double sumArray(const ImageContainer* image);
// void reduceImage( const ImageContainer* image, ImageContainer** imageOut, Vec<double> reductions);
// size_t* retrieveHistogram(const ImageContainer* image, int& returnSize);
// double* retrieveNormalizedHistogram(const ImageContainer* image, int& returnSize);
// void thresholdFilter(const ImageContainer* image, ImageContainer* imageOut, double threshold);
