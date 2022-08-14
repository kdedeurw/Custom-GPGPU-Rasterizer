#pragma once
//CUDA includes
#include "GPUHelpers.h"

GPU_CALLABLE GPU_INLINE static
void MultiplyMatrixMatrix(const float* matA, const float* matB, float* matC, unsigned int matSize)
{
	const unsigned int col = blockIdx.x * matSize + threadIdx.x;
	const unsigned int row = blockIdx.y * matSize + threadIdx.y;

	if (col < matSize && row < matSize)
	{
		float sum{};
		for (unsigned int i{}; i < matSize; ++i)
		{
			//matA goes from left to right in memory (row)
			//matB goes from top to bottom in memory (column)
			sum += matA[(threadIdx.y * matSize) + i] * matB[(i * matSize) + threadIdx.x];
		}
		matC[col + row * matSize] = sum;
	}
}

GPU_CALLABLE GPU_INLINE static
void MultiplyMatrixMatrixSync(const float* mat, float* ioMat, unsigned int matSize)
{
	const unsigned int col = blockIdx.x * matSize + threadIdx.x;
	const unsigned int row = blockIdx.y * matSize + threadIdx.y;

	if (col < matSize && row < matSize)
	{
		float sum{};
		for (unsigned int i{}; i < matSize; ++i)
		{
			//matA goes from left to right in memory (row)
			//matB goes from top to bottom in memory (column)
			sum += mat[(threadIdx.y * matSize) + i] * ioMat[(i * matSize) + threadIdx.x];
		}
		__syncthreads();
		ioMat[col + row * matSize] = sum;
	}
}

GPU_CALLABLE GPU_INLINE static
void MultiplyMatrixVector(const float* mat, const float* p, float* output, unsigned int matSize)
{
	const unsigned int idx = threadIdx.x % matSize;

	float sum{};
	for (unsigned int i{}; i < matSize; ++i)
	{
		//mat goes from left to right, every column per row
		//p goes from left to right, every column for 1 row
		sum += mat[(idx * matSize) + i] * p[threadIdx.x];
	}
	output[threadIdx.x] = sum;
}

GPU_CALLABLE GPU_INLINE static
void MultiplyMatrixVectorSync(const float* mat, float* ioP, unsigned int matSize)
{
	const unsigned int idx = threadIdx.x % matSize;

	float sum{};
	for (unsigned int i{}; i < matSize; ++i)
	{
		//mat goes from left to right, every column per row
		//p goes from left to right, every column for 1 row
		sum += mat[(idx * matSize) + i] * ioP[threadIdx.x];
	}
	__syncthreads();
	ioP[threadIdx.x] = sum;
}