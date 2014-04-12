#pragma once
#include "Vec.h"

double cSumArray(const unsigned char* imageIn, size_t n, int device=0);
double cSumArray(const unsigned int* imageIn, size_t n, int device=0);
double cSumArray(const int* imageIn, size_t n, int device=0);
double cSumArray(const float* imageIn, size_t n, int device=0);
double cSumArray(const double* imageIn, size_t n, int device=0);
