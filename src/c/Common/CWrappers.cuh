#pragma once
#include "Vec.h"

unsigned char* cAddConstant(const unsigned char* imageIn, Vec<size_t> dims, double additive, unsigned char** imageOut=NULL, int device=0);
unsigned int* cAddConstant(const unsigned int* imageIn, Vec<size_t> dims, double additive, unsigned int** imageOut=NULL, int device=0);
int* cAddConstant(const int* imageIn, Vec<size_t> dims, double additive, int** imageOut=NULL, int device=0);
float* cAddConstant(const float* imageIn, Vec<size_t> dims, double additive, float** imageOut=NULL, int device=0);
double* cAddConstant(const double* imageIn, Vec<size_t> dims, double additive, double** imageOut=NULL, int device=0);

unsigned char* cAddImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double additive,
							 unsigned char** imageOut=NULL, int device=0);
unsigned int* cAddImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double additive,
							unsigned int** imageOut=NULL, int device=0);
int* cAddImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double additive, int** imageOut=NULL, int device=0);
float* cAddImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double additive, float** imageOut=NULL, int device=0);
double* cAddImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double additive, double** imageOut=NULL,
					  int device=0);

unsigned char* cGaussianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned char** imageOut=NULL,
							   int device=0);
unsigned int* cGaussianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned int** imageOut=NULL, int device=0);
int* cGaussianFilter(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, int** imageOut=NULL, int device=0);
float* cGaussianFilter(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, float** imageOut=NULL, int device=0);
double* cGaussianFilter(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, double** imageOut=NULL, int device=0);

unsigned char* cImagePow(const unsigned char* imageIn, Vec<size_t> dims, double power, unsigned char** imageOut=NULL, int device=0);
unsigned int* cImagePow(const unsigned int* imageIn, Vec<size_t> dims, double power, unsigned int** imageOut=NULL, int device=0);
int* cImagePow(const int* imageIn, Vec<size_t> dims, double power, int** imageOut=NULL, int device=0);
float* cImagePow(const float* imageIn, Vec<size_t> dims, double power, float** imageOut=NULL, int device=0);
double* cImagePow(const double* imageIn, Vec<size_t> dims, double power, double** imageOut=NULL, int device=0);

unsigned char* cMedianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut=NULL,
							int device=0);
unsigned int* cMedianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut=NULL, int device=0);
int* cMedianFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut=NULL, int device=0);
float* cMedianFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut=NULL, int device=0);
double* cMedianFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut=NULL, int device=0);

unsigned char* cMultiplyImage(const unsigned char* imageIn, Vec<size_t> dims, double multiplier, unsigned char** imageOut=NULL, int device=0);
unsigned int* cMultiplyImage(const unsigned int* imageIn, Vec<size_t> dims, double multiplier, unsigned int** imageOut=NULL, int device=0);
int* cMultiplyImage(const int* imageIn, Vec<size_t> dims, double multiplier, int** imageOut=NULL, int device=0);
float* cMultiplyImage(const float* imageIn, Vec<size_t> dims, double multiplier, float** imageOut=NULL, int device=0);
double* cMultiplyImage(const double* imageIn, Vec<size_t> dims, double multiplier, double** imageOut=NULL, int device=0);

unsigned char* cMultiplyImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double factor,
								unsigned char** imageOut=NULL, int device=0);
unsigned int* cMultiplyImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double factor, unsigned int** imageOut=NULL, int device=0);
int* cMultiplyImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double factor, int** imageOut=NULL, int device=0);
float* cMultiplyImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double factor, float** imageOut=NULL, int device=0);
double* cMultiplyImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double factor, double** imageOut=NULL,
						 int device=0);

double cSumArray(const unsigned char* imageIn, size_t n, int device=0);
double cSumArray(const unsigned int* imageIn, size_t n, int device=0);
double cSumArray(const int* imageIn, size_t n, int device=0);
double cSumArray(const float* imageIn, size_t n, int device=0);
double cSumArray(const double* imageIn, size_t n, int device=0);
