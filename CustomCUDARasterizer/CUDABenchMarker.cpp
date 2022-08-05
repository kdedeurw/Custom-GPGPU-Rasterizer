#include "PCH.h"
#include "CUDABenchMarker.h"

CUDABenchMarker::CUDABenchMarker(int amountOfTimes, int amountOfTimeAverages)
	: m_AmountOfTimes{ amountOfTimes }
	, m_AmountOfTimeAverages{ amountOfTimeAverages }
	, m_TimesMs{}
	, m_StartEvent{}
	, m_StopEvent{}
{
	m_TimesMs = new float[amountOfTimes * amountOfTimeAverages];
}

CUDABenchMarker::~CUDABenchMarker()
{
	CheckErrorCuda(DeviceSynchroniseCuda());
	CheckErrorCuda(cudaEventDestroy(m_StartEvent));
	CheckErrorCuda(cudaEventDestroy(m_StopEvent));
	delete[] m_TimesMs;
}

void CUDABenchMarker::StartTimer()
{
	CheckErrorCuda(cudaEventRecord(m_StartEvent));
}

float CUDABenchMarker::StopTimer()
{
	float timerMs{};
	CheckErrorCuda(cudaEventRecord(m_StopEvent));
	CheckErrorCuda(cudaEventSynchronize(m_StopEvent));
	CheckErrorCuda(cudaEventElapsedTime(&timerMs, m_StartEvent, m_StopEvent));
	return timerMs;
}