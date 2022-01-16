#pragma once

namespace CUDASharedMemoryExtended
{
	__global__ void copy(float* odata, const float* idata);
	__global__ void transposeNaive(float* odata, const float* idata);
	__global__ void transposeCoalesced(float* odata, const float* idata);
	__global__ void copySharedMem(float* odata, const float* idata);
	__global__ void transposeNoBankConflicts(float* odata, const float* idata);
	cudaError_t SharedMemoryExtendedCuda();
}