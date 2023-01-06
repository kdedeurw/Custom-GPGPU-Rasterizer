#pragma once
#include "GPUHelpers.h"

namespace CUDAAtomicQueueOP
{
	template <typename T>
	GPU_CALLABLE GPU_INLINE static
	void Insert(T* dev_Queue, unsigned int& dev_QueueSize, int& dev_QueueSizeMutex, unsigned int queueMaxSize, T data)
	{
		bool isDone = false;
		do
		{
			isDone = (atomicCAS(&dev_QueueSizeMutex, 0, 1) == 0);
			if (isDone)
			{
				//critical section
				if (dev_QueueSize < queueMaxSize)
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
	//Only valid when the queue is NOT full
	void InsertUnsafe(T* dev_Queue, unsigned int& dev_QueueSize, int& dev_QueueSizeMutex, unsigned int queueMaxSize, T data)
	{
		bool isDone = false;
		do
		{
			isDone = (atomicCAS(&dev_QueueSizeMutex, 0, 1) == 0);
			if (isDone)
			{
				//critical section
				dev_Queue[dev_QueueSize] = data;
				++dev_QueueSize;
				dev_QueueSizeMutex = 0;
				//end of critical section
			}
		} while (!isDone);
	}

	template <typename T>
	GPU_CALLABLE GPU_INLINE static
	T Fetch(T* dev_Queue, unsigned int& dev_QueueSize, int& dev_QueueSizeMutex)
	{
		T data;
		bool isDone = false;
		do
		{
			isDone = (atomicCAS(&dev_QueueSizeMutex, 0, 1) == 0);
			if (isDone)
			{
				//critical section
				if (dev_QueueSize > 0)
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

	template <typename T>
	GPU_CALLABLE GPU_INLINE static
	//Only valid when the queue is NOT empty
	T FetchUnsafe(T* dev_Queue, unsigned int& dev_QueueSize, int& dev_QueueSizeMutex)
	{
		T data;
		bool isDone = false;
		do
		{
			isDone = (atomicCAS(&dev_QueueSizeMutex, 0, 1) == 0);
			if (isDone)
			{
				//critical section
				data = dev_Queue[dev_QueueSize];
				--dev_QueueSize;
				dev_QueueSizeMutex = 0;
				return data;
				//end of critical section
			}
		} while (!isDone);
		return data;
	}
}

template <typename T>
class CUDAAtomicQueue
{
public:
	CPU_CALLABLE
	virtual void Init(unsigned int queueMaxSize)
	{
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_Queue, queueMaxSize * sizeof(T)));
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_QueueSize, 4));
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_QueueSizeMutex, 4));
		m_QueueMaxSize = queueMaxSize;
	}

	CPU_CALLABLE
	virtual void Destroy()
	{
		CheckErrorCuda(cudaFree(m_Dev_QueueSizeMutex));
		CheckErrorCuda(cudaFree(m_Dev_QueueSize));
		CheckErrorCuda(cudaFree(m_Dev_Queue));
	}

	CPU_CALLABLE
	void ResetQueueSize()
	{
		CheckErrorCuda(cudaMemset(m_Dev_QueueSize, 0, 4));
	}

	CPU_CALLABLE
	void ResetQueue()
	{
		CheckErrorCuda(cudaMemset(m_Dev_Queue, 0, m_QueueMaxSize * sizeof(T)));
	}

	CPU_CALLABLE
	unsigned int GetQueueMaxSize() const { return m_QueueMaxSize; }
	CPU_CALLABLE
	unsigned int* GetDev_QueueSize() const { return m_Dev_QueueSize; }
	CPU_CALLABLE
	int* GetDev_QueueSizeMutexBuffer() const { return m_Dev_QueueSizeMutexBuffer; }
	CPU_CALLABLE
	T* GetDev_Queue() const { return m_Dev_Queue; }

protected:
	unsigned int m_QueueMaxSize;
	unsigned int* m_Dev_QueueSize;
	int* m_Dev_QueueSizeMutexBuffer;
	T* m_Dev_Queue;
};

template <typename T>
class CUDAAtomicQueues
{
public:
	CPU_CALLABLE
	virtual void Init(unsigned int numQueuesX, unsigned int numQueuesY, unsigned int queueMaxSize)
	{
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_Queues, numQueuesX * numQueuesY * queueMaxSize * sizeof(T)));
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_QueueSizes, numQueuesX * numQueuesY * 4));
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_QueueSizesMutex, numQueuesX * numQueuesY * 4));
		m_QueueMaxSize = queueMaxSize;
		m_NumQueuesX = numQueuesX;
		m_NumQueuesY = numQueuesY;
	}

	CPU_CALLABLE
	virtual void Destroy()
	{
		CheckErrorCuda(cudaFree(m_Dev_QueueSizesMutex));
		CheckErrorCuda(cudaFree(m_Dev_QueueSizes));
		CheckErrorCuda(cudaFree(m_Dev_Queues));
	}

	CPU_CALLABLE
	void ResetQueueSizes()
	{
		const unsigned int size = m_NumQueuesX * m_NumQueuesY * 4;
		CheckErrorCuda(cudaMemset(m_Dev_QueueSizes, 0, size));
	}

	CPU_CALLABLE
	void ResetQueues()
	{
		const unsigned int size = m_NumQueuesX * m_NumQueuesY * m_QueueMaxSize * sizeof(T);
		CheckErrorCuda(cudaMemset(m_Dev_Queues, 0, size));
	}

	CPU_CALLABLE
	unsigned int GetQueueMaxSize() const { return m_QueueMaxSize; }
	CPU_CALLABLE
	unsigned int GetNumQueuesX() const { return m_NumQueuesX; }
	CPU_CALLABLE
	unsigned int GetNumQueuesY() const { return m_NumQueuesY; }
	CPU_CALLABLE
	unsigned int* GetDev_QueueSizes() const { return m_Dev_QueueSizes; }
	CPU_CALLABLE
	int* GetDev_QueueSizesMutexBuffer() const { return m_Dev_QueueSizesMutex; }
	CPU_CALLABLE
	T* GetDev_Queues() const { return m_Dev_Queues; }

	CPU_CALLABLE
	unsigned int* GetDev_QueueSizes(unsigned int idx) const { return &m_Dev_QueueSizes[idx * m_NumQueuesX * m_NumQueuesY]; }
	CPU_CALLABLE
	int* GetDev_QueueSizesMutexBuffer(unsigned int idx) const { return &m_Dev_QueueSizesMutex[idx * m_NumQueuesX * m_NumQueuesY]; }
	CPU_CALLABLE
	T* GetDev_Queue(unsigned int idx) const { return &m_Dev_Queues[idx * m_NumQueuesX * m_NumQueuesY * m_QueueMaxSize]; }

protected:
	unsigned int m_QueueMaxSize;
	unsigned int m_NumQueuesX;
	unsigned int m_NumQueuesY;
	unsigned int* m_Dev_QueueSizes;
	int* m_Dev_QueueSizesMutex;
	T* m_Dev_Queues;
};

//template <typename T>
//struct CUDAAtomicQueueCompact
//{
//	unsigned int* dev_QueueSize;
//	int* dev_QueueSizeMutexBuffer;
//	T* dev_Queue;
//};

//template <typename T>
//struct CUDAAtomicQueueCompactSize
//{
//	unsigned int QueueMaxSize;
//	unsigned int* dev_QueueSize;
//	int* dev_QueueSizeMutexBuffer;
//	T* dev_Queue;
//};