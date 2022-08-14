#pragma once
#include "GPUHelpers.h"

class CUDABenchMarker
{
public:
	CPU_CALLABLE CUDABenchMarker(int amountOfTimes = 4, int amountOfTimeAverages = 10);
	CPU_CALLABLE virtual ~CUDABenchMarker();

	CPU_CALLABLE virtual void StartTimer();
	CPU_CALLABLE virtual float StopTimer();

private:
	int m_AmountOfTimes, m_AmountOfTimeAverages;
	float* m_TimesMs;
	cudaEvent_t m_StartEvent, m_StopEvent;
};