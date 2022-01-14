
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>

#include "Matrix.h"
#include "CUDASharedMemory.cuh"

//Choose which GPU to run on, change this on a multi-GPU system. (Default is 0, for single-GPU systems)
cudaError_t SetDeviceCuda(int deviceId = 0);
//Calls cudaGetLastError, this checks for any errors launching the kernel
cudaError_t CheckErrorCuda();
//Calls cudaDeviceSynchronize, this waits for the kernel to finish, and returns any errors encountered during the launch.
cudaError_t DeviceSynchroniseCuda();
//Same as DeviceSynchroniseCuda, DEPRECATED
//cudaError_t ThreadSynchroniseCuda();
//Calls cudaDeviceReset, this must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
cudaError_t DeviceResetCuda();

__global__ void VecAddKernel(int *c, const int *a, const int *b)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
//                          float C[X][Y],  float A[X][Y],  float B[X][Y]
__global__ void MatAdd2DKernel(   float* C,       float* A,       float* B, const int matSizeX, const int matSizeY)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < matSizeX && j < matSizeY)
        C[i + matSizeX * j] = A[i + matSizeX * j] + B[i + matSizeX * j];
}
//                          float C[N*N],   float A[N*N],   float B[N*N]
__global__ void MatAdd1DKernel(   float* C,       float* A,       float* B, const int matSize)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < matSize * matSize)
        C[i] = A[i] + B[i];
}

__global__ void AddDifferentKernel(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix<float> A, Matrix<float> B, Matrix<float> C)
{
//    // Block row and column
//    const int blockRow = blockIdx.y;
//    const int blockCol = blockIdx.x;
//
//    // Each thread block computes one sub-matrix Csub of C
//    Matrix<float> Csub = GetSubMatrix(C, blockRow, blockCol);
//
//    // Each thread computes one element of Csub
//    // by accumulating results into Cvalue
//    float Cvalue = 0;
//
//    // Thread row and column within Csub
//    int row = threadIdx.y;
//    int col = threadIdx.x;
//
//    // Loop over all the sub-matrices of A and B that are
//    // required to compute Csub
//    // Multiply each pair of sub-matrices together
//    // and accumulate the results
//    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
//
//        // Get sub-matrix Asub of A
//        Matrix Asub = GetSubMatrix(A, blockRow, m);
//
//        // Get sub-matrix Bsub of B
//        Matrix Bsub = GetSubMatrix(B, m, blockCol);
//
//        // Shared memory used to store Asub and Bsub respectively
//        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
//        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
//
//        // Load Asub and Bsub from device memory to shared memory
//        // Each thread loads one element of each sub-matrix
//        As[row][col] = GetElement(Asub, row, col);
//        Bs[row][col] = GetElement(Bsub, row, col);
//
//        // Synchronize to make sure the sub-matrices are loaded
//        // before starting the computation
//        __syncthreads();
//        // Multiply Asub and Bsub together
//        for (int e = 0; e < BLOCK_SIZE; ++e)
//            Cvalue += As[row][e] * Bs[e][col];
//
//        // Synchronize to make sure that the preceding
//        // computation is done before loading two new
//        // sub-matrices of A and B in the next iteration
//        __syncthreads();
//    }
//
//    // Write Csub to device memory
//    // Each thread writes one element
//    SetElement(Csub, row, col, Cvalue);
}

cudaError_t VecAddCuda(int* c, const int* a, const int* b, unsigned int size);
cudaError_t MatAddCuda(float* matC, const float* matA, const float* matB, unsigned int matDimX);
int main()
{
    cudaError_t cudaStatus{};
    cudaStatus = SetDeviceCuda();

    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };
    //
    //cudaStatus = VecAddCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "VecAddCuda failed!");
    //    return 1;
    //}
    //
    //cudaStatus = DeviceResetCuda();
    //
    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);
    //
    ////MatAdd
    //const int N = 2;
    //float matA[N * N]{ 1.f, 2.f, 3.f, 4.f };
    //float matB[N * N]{ 4.f, 3.f, 2.f, 1.f };
    //float matC[N * N]{ -1.f, -1.f, -1.f, -1.f };
    //
    //cudaStatus = MatAddCudaSharedMemory(matC, matA, matB, N);
    //if (cudaStatus != cudaSuccess)
    //{
    //    fprintf(stderr, "MatAddCudaSharedMemory failed!");
    //    return 1;
    //}
    //
    //cudaStatus = DeviceResetCuda();
    //
    //printf("{1.f,2.f,3.f,4.f} + {4.f,3.f,2.f,1.f} = {%f,%f,%f,%f}\n", 
    //    matC[0], matC[1], matC[2], matC[3]);

    int statusCode = CUDASharedMemory::SharedMemoryCuda();

    return statusCode;
}

cudaError_t SetDeviceCuda(int deviceId)
{
    cudaError_t cudaStatus = cudaSetDevice(deviceId);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    return cudaStatus;
}

cudaError_t CheckErrorCuda()
{
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "latest kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    return cudaStatus;
}

cudaError_t DeviceSynchroniseCuda()
{
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
    }
    return cudaStatus;
}

//DEPRECATED
//cudaError_t ThreadSynchroniseCuda()
//{
//    cudaError_t cudaStatus = cudaThreadSynchronize();
//    if (cudaStatus != cudaSuccess)
//    {
//        fprintf(stderr, "cudaThreadSynchronize returned error code %d !\n", cudaStatus);
//    }
//    return cudaStatus;
//}

cudaError_t DeviceResetCuda()
{
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t VecAddCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
    const int numBlocks = 1;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    VecAddKernel<<<numBlocks, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = CheckErrorCuda();
    if (cudaStatus != cudaSuccess)
        goto Error;
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = DeviceSynchroniseCuda();
    if (cudaStatus != cudaSuccess)
        goto Error;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t MatAddCuda(float* matC, const float* matA, const float* matB, unsigned int matDim)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    const unsigned int matSize = matDim * matDim;
    //Allocate GPU memory
    cudaStatus = cudaMalloc((void**)&dev_c, matSize * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, matSize * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_b, matSize * sizeof(float));

    //Copy CPU memory to GPU
    cudaStatus = cudaMemcpy(dev_a, matA, matSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, matB, matSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_c, matC, matSize * sizeof(float), cudaMemcpyHostToDevice);
     
    //Kernel invocation with blocks of 16x16 (256 threads)
    const dim3 threadsPerBlock{ 16, 16 };
    //numBlocks = (N + blockSize - 1) / blockSize;
    const int N = matSize; //computational size
    const dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatAdd2DKernel<<<numBlocks, threadsPerBlock>>> (dev_c, dev_a, dev_b, matDim, matDim);

    //Check for errors
    cudaStatus = CheckErrorCuda();
    if (cudaStatus != cudaSuccess)
        goto Error;

    //Synchronise Device
    cudaStatus = DeviceSynchroniseCuda();
    if (cudaStatus != cudaSuccess)
        goto Error;

    //Copy GPU memory to CPU
    cudaStatus = cudaMemcpy(matC, dev_c, matSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    //Deallocate GPU memory
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

__constant__ float constData[256];
__device__ float devData;
__device__ float* devPointer;
cudaError_t ConstDataAccesses()
{
    cudaError_t cudaStatus{};

    float data[256];
    //Copy data to constData
    cudaStatus = cudaMemcpyToSymbol(constData, data, sizeof(data));
    //Copy constData to data
    cudaStatus = cudaMemcpyFromSymbol(data, constData, sizeof(data));

    float value = 3.14f;
    cudaStatus = cudaMemcpyToSymbol((void**)&devData, &value, sizeof(float));

    float* ptr;
    cudaStatus = cudaMalloc(&ptr, 256 * sizeof(float));
    cudaStatus = cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

    return cudaStatus;
}

cudaError_t L2CacheSetAside()
{
    cudaError_t cudaError{};
    cudaDeviceProp prop;
    cudaError = cudaGetDeviceProperties(&prop, 0); //id == 0, for single GPU systems
    size_t size = std::min(int(prop.l2CacheSize * 0.75f), prop.persistingL2CacheMaxSize);
    cudaError = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
    return cudaError;
}

cudaError_t CUDAStreamExample()
{
    const size_t numBytes{};
    void* ptr{};
    cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = numBytes;                    // Number of bytes for persistence access.
                                                                                  // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    stream_attribute.accessPolicyWindow.hitRatio = 0.6f;                          // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    //Set the attributes to a CUDA stream of type cudaStream_t
    cudaStream_t stream{};
    return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}

cudaError_t CUDAGraphKernelNodeExample()
{
    const size_t numBytes{};
    void* ptr{};
    cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
    node_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr); // Global Memory data pointer
    node_attribute.accessPolicyWindow.num_bytes = numBytes;                    // Number of bytes for persistence access.
                                                                                // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    node_attribute.accessPolicyWindow.hitRatio = 0.6f;                          // Hint for cache hit ratio
    node_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
    node_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

    //Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
    cudaGraphNode_t node{};
    return cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
}

void L2CachePersistence()
{
    //cudaStream_t stream;
    //cudaStreamCreate(&stream);                                                                  // Create CUDA stream
    //const size_t numBytes{};
    //void* ptr{};
    //
    //cudaDeviceProp prop;                                                                        // CUDA device properties variable
    //cudaGetDeviceProperties(&prop, 0);                                                 // Query GPU properties
    //size_t size = std::min(int(prop.l2CacheSize * 0.75f), prop.persistingL2CacheMaxSize);
    //cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed
    //
    //size_t window_size = std::min((size_t)prop.accessPolicyMaxWindowSize, numBytes);                        // Select minimum of user defined num_bytes and max window size.
    //
    //cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
    //stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(data1);               // Global Memory data pointer
    //stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
    //stream_attribute.accessPolicyWindow.hitRatio = 0.6f;                                        // Hint for cache hit ratio
    //stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;               // Persistence Property
    //stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;                // Type of access property on cache miss
    //
    //cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream
    //
    //for (int i = 0; i < 10; i++)
    //{
    //    cuda_kernelA<<<grid_size, block_size, 0, stream>>>(data1);                                 // This data1 is used by a kernel multiple times
    //}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
    //cuda_kernelB<<<grid_size, block_size, 0, stream>>>(data1);                                     // A different kernel in the same stream can also benefit
    //                                                                                            // from the persistence of data1
    //
    //stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
    //cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
    //cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2 
    //
    //cuda_kernelC<<<grid_size, block_size, 0, stream>>>(data2);                                     // data2 can now benefit from full L2 in normal mode
}