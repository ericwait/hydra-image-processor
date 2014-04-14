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

unsigned char* cImagePow(const unsigned char* imageIn, Vec<size_t> dims, double power, unsigned char** imageOut=NULL, int device=0);
unsigned int* cImagePow(const unsigned int* imageIn, Vec<size_t> dims, double power, unsigned int** imageOut=NULL, int device=0);
int* cImagePow(const int* imageIn, Vec<size_t> dims, double power, int** imageOut=NULL, int device=0);
float* cImagePow(const float* imageIn, Vec<size_t> dims, double power, float** imageOut=NULL, int device=0);
double* cImagePow(const double* imageIn, Vec<size_t> dims, double power, double** imageOut=NULL, int device=0);

double cSumArray(const unsigned char* imageIn, size_t n, int device=0);
double cSumArray(const unsigned int* imageIn, size_t n, int device=0);
double cSumArray(const int* imageIn, size_t n, int device=0);
double cSumArray(const float* imageIn, size_t n, int device=0);
double cSumArray(const double* imageIn, size_t n, int device=0);
