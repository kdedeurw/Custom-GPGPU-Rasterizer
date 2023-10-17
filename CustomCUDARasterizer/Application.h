#pragma once
#include "ETimer.h"
#include "Camera.h"
#include "SceneManager.h"
#include "CUDASceneGraph.h"
#include "CUDATextureManager.h"
#include "WindowHelper.h"

struct SDL_Window;
class SceneManager;
class CUDARenderer;
class CUDATextureManager;
class Application final
{
public:
	Application(Camera camera);
	~Application();

	//deviceId = 0 for single GPU setup
	bool Init(int deviceId = 0, Resolution::ResolutionStandard rs = Resolution::ResolutionStandard::SD);
	void Run();
	void LoadSceneGraph(SceneGraph* pSceneGraph);
	CUDATextureManager& GetTextureManager() { return m_TextureManager; }

private:
	SDL_Window* m_pWindow;
	CUDARenderer* m_pRenderer;
	WindowHelper m_WindowHelper;
	Elite::Timer m_Timer;
	Camera m_Camera;
	CUDASceneGraph m_SceneGraph;
	CUDATextureManager m_TextureManager;

	void DisplayResolutionDetails(const Resolution& res);
	void CUDACheckBankConflicts(unsigned int dataSizePerThread);
	int GetFPSImmediate(float ms);
	void UpdateCamera(float elapsedSec);
	bool CheckBinMACROS(const Resolution& res, int& binMultiplier, IPoint2& numBins, IPoint2& binDim, int& binQueueMaxSize);
	void PrintInfo();
	void Shutdown();
};