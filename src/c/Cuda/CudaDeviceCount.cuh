#pragma once
#include "cuda_runtime_api.h"

int cDeviceCount()
{
	int cnt = 0;
	cudaGetDeviceCount(&cnt);

	return cnt;
}