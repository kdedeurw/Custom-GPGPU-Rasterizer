#include "PCH.h"

namespace CUDAStreams
{
	cudaError_t StreamTestDONTCALL()
	{
		cudaError_t cudaStatus{};
		const unsigned int N = 1024;
		float h_a[N]{};
		float* d_a{};

		//Synchronised calls
		cudaMemcpy(d_a, h_a, N, cudaMemcpyHostToDevice); //CPU will wait until this transfer is complete
		//(asynchronous) increment<<<1, N>>>(d_a)
		//myCpuFunction(b) (asynchronous && independent CPU code)
		cudaMemcpy(h_a, d_a, N, cudaMemcpyDeviceToHost); //CPU will wait until this transfer is complete

		//Create stream
		cudaStream_t stream1;
		cudaError_t result;
		result = cudaStreamCreate(&stream1);
		result = cudaStreamDestroy(stream1);

		//Asynchronised calls aka NON-BLOCKING the host
		result = cudaMemcpyAsync(d_a, h_a, N, cudaMemcpyHostToDevice, stream1);

		//increment<<<1, N, 0, stream1>>>(d_a)

		return cudaStatus;
	}

	__global__ void kernel(float* a, int offset)
	{
		int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
		float x = (float)i;
		float s = sinf(x);
		float c = cosf(x);
		a[i] = a[i] + sqrtf(s * s + c * c);
	}

	float maxError(float* a, int n)
	{
		float maxE = 0;
		for (int i = 0; i < n; i++)
		{
			float error = fabs(a[i] - 1.0f);
			if (error > maxE)
				maxE = error;
		}
		return maxE;
	}

	cudaError_t StreamCuda()
	{
		cudaError_t cudaStatus{};

		const int blockSize = 256, nStreams = 4;
		const int n = 4 * 1024 * blockSize * nStreams; //16MBs * 4 streams
		const int streamSize = n / nStreams; //16MBs
		const int nStreamBytes = streamSize * sizeof(float); //16MBs * 4
		const int nBytes = n * sizeof(float); //total amount of bytes

		cudaDeviceProp prop;
		checkCuda(cudaGetDeviceProperties(&prop, 0));
		printf("Device : %s\n", prop.name);
		checkCuda(cudaSetDevice(0));

		// allocate pinned host memory and device memory
		float* h_a, * d_a;
		checkCuda(cudaMallocHost((void**)&h_a, nBytes));      // host pinned
		checkCuda(cudaMalloc((void**)&d_a, nBytes)); // device

		float ms; // elapsed time in milliseconds

		// create events and streams
		cudaEvent_t startEvent, stopEvent, dummyEvent;
		cudaStream_t stream[nStreams];
		checkCuda(cudaEventCreate(&startEvent));
		checkCuda(cudaEventCreate(&stopEvent));
		checkCuda(cudaEventCreate(&dummyEvent));
		for (int i = 0; i < nStreams; ++i)
		{
			checkCuda(cudaStreamCreate(&stream[i]));
		}

		// baseline case - sequential transfer and execute
		memset(h_a, 0, nBytes);
		checkCuda(cudaEventRecord(startEvent, 0));
		checkCuda(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));

		//		NumBlocks,		ThreadsPerBlock
		kernel<<<n / blockSize, blockSize>>>(d_a, 0);

		checkCuda(cudaMemcpy(h_a, d_a, nBytes, cudaMemcpyDeviceToHost));
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));

		printf("Time for sequential transfer and execute (ms): %f\n", ms);
		printf("  max error: %e\n", maxError(h_a, n));

		// asynchronous version 1: loop over {copy, kernel, copy}
		memset(h_a, 0, nBytes);
		checkCuda(cudaEventRecord(startEvent, 0));

		for (int i = 0; i < nStreams; ++i)
		{
			int offset = i * streamSize;
			checkCuda(cudaMemcpyAsync(&d_a[offset], &h_a[offset], nStreamBytes, cudaMemcpyHostToDevice, stream[i]));

			kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);

			checkCuda(cudaMemcpyAsync(&h_a[offset], &d_a[offset], nStreamBytes, cudaMemcpyDeviceToHost,stream[i]));
		}

		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
		printf("  max error: %e\n", maxError(h_a, n));

		// asynchronous version 2: 
		// loop over copy, loop over kernel, loop over copy
		memset(h_a, 0, nBytes);
		checkCuda(cudaEventRecord(startEvent, 0));

		for (int i = 0; i < nStreams; ++i)
		{
			int offset = i * streamSize;
			checkCuda(cudaMemcpyAsync(&d_a[offset], &h_a[offset], nStreamBytes, cudaMemcpyHostToDevice, stream[i]));
		}

		for (int i = 0; i < nStreams; ++i)
		{
			int offset = i * streamSize;
			kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
		}

		for (int i = 0; i < nStreams; ++i)
		{
			int offset = i * streamSize;
			checkCuda(cudaMemcpyAsync(&h_a[offset], &d_a[offset], nStreamBytes, cudaMemcpyDeviceToHost,stream[i]));
		}

		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
		printf("  max error: %e\n", maxError(h_a, n));

		// cleanup
		checkCuda(cudaEventDestroy(startEvent));
		checkCuda(cudaEventDestroy(stopEvent));
		checkCuda(cudaEventDestroy(dummyEvent));

		for (int i = 0; i < nStreams; ++i)
		{
			checkCuda(cudaStreamDestroy(stream[i]));
		}

		cudaFree(d_a);
		cudaFreeHost(h_a);

		return cudaStatus;
	}
}