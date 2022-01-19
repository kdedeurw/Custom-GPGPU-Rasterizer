/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// GPUHelpers.h: CUDA defines
/*=============================================================================*/
#pragma once
#pragma warning(push)
#pragma warning(disable: 26812)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning(pop)

#include <stdio.h>
#include <iostream>

#define GPUErrorCheck(ans){GPUAssert((ans), __FILE__, __LINE__);}
inline void GPUAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUAssert: %s %s %i \n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#define GPU_CALLABLE	__device__
#define CPU_CALLABLE	__host__
#define BOTH_CALLABLE	__host__ __device__
//Own evil MACRO's
#define GPU_KERNEL __global__
#define GPU_SHARED_MEMORY __shared__