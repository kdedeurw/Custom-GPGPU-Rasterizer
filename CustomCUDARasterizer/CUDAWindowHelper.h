#pragma once
#include "GPUHelpers.h"

template <typename T>
class CUDAWindowHelper
{
public:
	CPU_CALLABLE
	virtual void Init(unsigned int width, unsigned int height)
	{
		unsigned int size = sizeof(T);
		CheckErrorCuda(cudaFree(m_Dev_FragmentBuffer));
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_FragmentBuffer, width * height * size));
		CheckErrorCuda(cudaMemset(m_Dev_FragmentBuffer, 0, width * height * size));

		//The framebuffer in device memory
		size = sizeof(unsigned int);
		CheckErrorCuda(cudaFree(m_Dev_FrameBuffer));
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_FrameBuffer, width * height * size));
		CheckErrorCuda(cudaMemset(m_Dev_FrameBuffer, 0, width * height * size));

		size = sizeof(int);
		CheckErrorCuda(cudaFree(m_Dev_DepthBuffer));
		CheckErrorCuda(cudaMalloc((void**)&m_Dev_DepthBuffer, width * height * size));
		CheckErrorCuda(cudaMemset(m_Dev_DepthBuffer, 0, width * height * size));

		size = sizeof(int);
		cudaFree(m_Dev_DepthMutexBuffer);
		cudaMalloc((void**)&m_Dev_DepthMutexBuffer, width * height * size);
		cudaMemset(m_Dev_DepthMutexBuffer, 0, width * height * size);
	}

	CPU_CALLABLE
	virtual void Destroy()
	{
		CheckErrorCuda(cudaFree(m_Dev_DepthMutexBuffer));
		m_Dev_DepthMutexBuffer = nullptr;

		CheckErrorCuda(cudaFree(m_Dev_DepthBuffer));
		m_Dev_DepthBuffer = nullptr;

		CheckErrorCuda(cudaFree(m_Dev_FrameBuffer));
		m_Dev_FrameBuffer = nullptr;

		CheckErrorCuda(cudaFree(m_Dev_FragmentBuffer));
		m_Dev_FragmentBuffer = nullptr;
	}

	CPU_CALLABLE
	unsigned int* GetDev_FrameBuffer() const { return m_Dev_FrameBuffer; }
	CPU_CALLABLE
	int* GetDev_DepthBuffer() const { return m_Dev_DepthBuffer; }
	CPU_CALLABLE
	int* GetDev_DepthMutexBuffer() const { return m_Dev_DepthMutexBuffer; }
	CPU_CALLABLE
	T* GetDev_FragmentBuffer() const { return m_Dev_FragmentBuffer; }

protected:
	unsigned int* m_Dev_FrameBuffer;
	int* m_Dev_DepthBuffer; 
	int* m_Dev_DepthMutexBuffer;
	T* m_Dev_FragmentBuffer;
};