#pragma once
//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Standard includes
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <iostream>

//Math includes
#include "Math.h"
#include "MathUtilities.h"
#include "RGBColor.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t CheckErrorCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

//Choose which GPU to run on, change this on a multi-GPU system. (Default is 0, for single-GPU systems)
inline cudaError_t SetDeviceCuda(int deviceId = 0)
{
    cudaError_t cudaStatus = cudaSetDevice(deviceId);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    return cudaStatus;
}

//Calls cudaDeviceSynchronize, this waits for the kernel to finish, and returns any errors encountered during the launch.
inline cudaError_t DeviceSynchroniseCuda()
{
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
    }
    return cudaStatus;
}

//Calls cudaDeviceReset, this must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
inline cudaError_t DeviceResetCuda()
{
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    return cudaStatus;
}