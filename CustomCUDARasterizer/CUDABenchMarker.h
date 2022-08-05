#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

class CUDABenchMarker
{
public:
	CPU_CALLABLE CUDABenchMarker(int amountOfTimes = 4, int amountOfTimeAverages = 10);
	CPU_CALLABLE ~CUDABenchMarker();

	CPU_CALLABLE void StartTimer();
	CPU_CALLABLE float StopTimer();

private:
	int m_AmountOfTimes, m_AmountOfTimeAverages;
	float* m_TimesMs;
	cudaEvent_t m_StartEvent, m_StopEvent;
};