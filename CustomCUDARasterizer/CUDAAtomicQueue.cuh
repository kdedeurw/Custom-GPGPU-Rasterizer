#pragma once
#include "GPUHelpers.h"

namespace CUDAAtomicQueueOperations
{
	template <typename T>
	GPU_CALLABLE GPU_INLINE static
	void Insert(T* dev_Queue, unsigned int* dev_QueueSize, int* dev_QueueSizeMutex,
		unsigned int queueMaxSize, T data)
	{
		bool isDone = false;
		do
		{
			isDone = (atomicCAS(dev_QueueSizeMutex, 0, 1) == 0);
			if (isDone)
			{
				//critical section
				if (*dev_QueueSize < queueMaxSize)
				{
					dev_Queue[dev_QueueSize] = data;
					++dev_QueueSize;
				}
				dev_QueueSizeMutex = 0;
				//end of critical section
			}
		} while (!isDone);
	}

	template <typename T>
	GPU_CALLABLE GPU_INLINE static
	T Fetch(T* dev_Queue, unsigned int* dev_QueueSize, int* dev_QueueSizeMutex)
	{
		T data;
		bool isDone = false;
		do
		{
			isDone = (atomicCAS(dev_QueueSizeMutex, 0, 1) == 0);
			if (isDone)
			{
				//critical section
				if (*dev_QueueSize > 0)
				{
					data = dev_Queue[dev_QueueSize];
					--dev_QueueSize;
				}
				dev_QueueSizeMutex = 0;
				//end of critical section
			}
		} while (!isDone);
		return data;
	}
}

template <typename T>
struct CUDAAtomicQueue
{
	CPU_CALLABLE
	CUDAAtomicQueue(unsigned int queueMaxSize)
		: dev_Queue{}
		, dev_QueueSizeMutex{}
		, dev_QueueSize{}
		, QueueMaxSize{ queueMaxSize }
	{
		CheckErrorCuda(cudaMalloc((void**)&dev_Queue, queueMaxSize * sizeof(T)));
		CheckErrorCuda(cudaMalloc((void**)&dev_QueueSize, 4));
		CheckErrorCuda(cudaMalloc((void**)&dev_QueueSizeMutex, 4));
	}

	CPU_CALLABLE
	~CUDAAtomicQueue()
	{
		CheckErrorCuda(cudaFree(dev_QueueSizeMutex));
		CheckErrorCuda(cudaFree(dev_QueueSize));
		CheckErrorCuda(cudaFree(dev_Queue));
	}

	const unsigned int QueueMaxSize;
	unsigned int* dev_QueueSize;
	unsigned int* dev_QueueSizeMutex;
	T* dev_Queue;
};

template <typename T>
struct CUDAAtomicQueues
{
	CPU_CALLABLE
	CUDAAtomicQueues(unsigned int numQueuesX, unsigned int numQueuesY, unsigned int queueMaxSize)
		: dev_Queues{}
		, dev_QueueSizesMutex{}
		, dev_QueueSizes{}
		, QueueMaxSize{ queueMaxSize }
		, NumQueuesX{ numQueuesX }
		, NumQueuesY{ numQueuesY }
	{
		CheckErrorCuda(cudaMalloc((void**)&dev_Queues, numQueuesX * numQueuesY * queueMaxSize * sizeof(T)));
		CheckErrorCuda(cudaMalloc((void**)&dev_QueueSizes, numQueuesX * numQueuesY * 4));
		CheckErrorCuda(cudaMalloc((void**)&dev_QueueSizesMutex, numQueuesX * numQueuesY * 4));
	}

	CPU_CALLABLE
	~CUDAAtomicQueues()
	{
		CheckErrorCuda(cudaFree(dev_QueueSizesMutex));
		CheckErrorCuda(cudaFree(dev_QueueSizes));
		CheckErrorCuda(cudaFree(dev_Queues));
	}

	const unsigned int QueueMaxSize;
	const unsigned int NumQueuesX;
	const unsigned int NumQueuesY;
	unsigned int* dev_QueueSizes;
	unsigned int* dev_QueueSizesMutex;
	T* dev_Queues;
};