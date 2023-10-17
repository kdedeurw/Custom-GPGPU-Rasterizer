#include "PCH.h"
#include "Application.h"
#include "DEFINES.h"

//Project includes
#include "ETimer.h"
#include "EventManager.h"
#include "DirectionalLight.h"

//Project CUDA includes
#include "CUDARenderer.h"
#include "CUDATexture.h"
#include "CUDATextureManager.h"

//#include <curand_kernel.h>
//#include <curand.h>

Application::Application(Camera camera)
	: m_pWindow{}
	, m_pRenderer{}
	, m_WindowHelper{}
	, m_Timer{}
	, m_Camera{ std::move(camera) }
	, m_TextureManager{}
{
}

Application::~Application()
{
	Shutdown();
}

void Application::Shutdown()
{
	//Shutdown framework
	CheckErrorCuda(DeviceSynchroniseCuda());
	if (m_pRenderer)
		delete m_pRenderer;
	CheckErrorCuda(DeviceResetCuda());

	if (m_pWindow)
		SDL_DestroyWindow(m_pWindow);
	SDL_Quit();
}

void Application::UpdateCamera(float elapsedSec)
{
	const float moveSpeed = m_Camera.GetMoveSpeed();
	const float rotSpeed = m_Camera.GetRotationSpeed();
	const MouseInformation mi = EventManager::GetMouseInformation();

	if (mi.scrollwheel != 0)
	{
		m_Camera.SetMoveSpeed(moveSpeed + (float)mi.scrollwheel);
		m_Camera.SetRotationSpeed(rotSpeed + (mi.scrollwheel / 10.f));
	}

	if (mi.lmb && mi.rmb)
	{
		m_Camera.TranslateY(mi.y * moveSpeed * elapsedSec);
	}
	else if (mi.lmb && !mi.rmb)
	{
		m_Camera.TranslateZ(mi.y * moveSpeed * elapsedSec);
		m_Camera.Yaw(mi.x * rotSpeed);
	}
	else if (!mi.lmb && mi.rmb)
	{
		m_Camera.Yaw(-mi.x * rotSpeed);
		m_Camera.Pitch(-mi.y * rotSpeed);
	}

	if (EventManager::IsKeyDown(SDL_SCANCODE_W))
	{
		m_Camera.TranslateZ(-moveSpeed * elapsedSec);
	}
	else if (EventManager::IsKeyDown(SDL_SCANCODE_S))
	{
		m_Camera.TranslateZ(moveSpeed * elapsedSec);
	}
	if (EventManager::IsKeyDown(SDL_SCANCODE_A))
	{
		m_Camera.TranslateX(-moveSpeed * elapsedSec);
	}
	else if (EventManager::IsKeyDown(SDL_SCANCODE_D))
	{
		m_Camera.TranslateX(moveSpeed * elapsedSec);
	}
}

int Application::GetFPSImmediate(float ms)
{
	return int(1 / ms * 1000);
}

void Application::CUDACheckBankConflicts(unsigned int dataSizePerThread)
{
	int* memoryLayout = new int[32 * dataSizePerThread]{};

	for (int i{}; i < 32; ++i)
	{
		memoryLayout[i * dataSizePerThread] = 1;
	}

	for (int i{}; i < dataSizePerThread; ++i)
	{
		for (int j{}; j < 32; ++j)
		{
			std::cout << memoryLayout[i * 32 + j] << "|";
		}
		std::cout << '\n';
	}

	std::cout << "\nData size : " << dataSizePerThread << '\n';
	int totalBankConflicts{};
	for (int i{}; i < 32; ++i)
	{
		int bankConflicts{};
		for (int j{}; j < dataSizePerThread; ++j)
		{
			if (memoryLayout[i + j * 32] == 1)
			{
				++bankConflicts;
			}
		}
		if (bankConflicts > 1)
		{
			std::cout << bankConflicts << "-way conflicts at bank: " << i << '\n';
			++totalBankConflicts;
		}
	}
	if (totalBankConflicts > 0)
	{
		std::cout << "Total bank conflicts: " << totalBankConflicts << '\n';
	}
	else
	{
		std::cout << "No bank conflicts detected!\n";
	}
	delete[] memoryLayout;
}

void Application::DisplayResolutionDetails(const Resolution& res)
{
	std::cout << "------------------------------\n";
	std::cout << "Window size: " << res.Width << 'x' << res.Height << "p\n";
	std::cout << "AspectRatio: " << res.AspectRatio.w << ':' << res.AspectRatio.h << '\n';
	std::cout << "ResolutionStandard: ";
	switch (res.Standard)
	{
	case Resolution::VGA:
		std::cout << "VGA";
		break;
	case Resolution::SD:
		std::cout << "SD";
		break;
	case Resolution::HD:
		std::cout << "HD";
		break;
	case Resolution::FHD:
		std::cout << "FHD";
		break;
	case Resolution::QHD:
		std::cout << "QHD";
		break;
	case Resolution::UHD:
		std::cout << "UHD";
		break;
	default:
		break;
	}
	std::cout << '\n';
	std::cout << "------------------------------\n";
}

bool Application::Init(int deviceId, Resolution::ResolutionStandard rs)
{
	if (CheckErrorCuda(SetDeviceCuda(deviceId)) != cudaError_t::cudaSuccess)
		return false;

	//Create window + surfaces
	int errorCode = SDL_Init(SDL_INIT_VIDEO);
	if (errorCode != 0)
	{
		std::cout << "SDL failed to initialize: " << SDL_GetError() << '\n';
		return false;
	}

	//Select resolution
	const Resolution res = Resolution::GetResolution(rs);

	m_pWindow = SDL_CreateWindow(
		"Custom GPGPU CUDA Rasterizer - GW Kristof Dedeurwaerder 2022",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		res.Width, res.Height, 0);

	if (!m_pWindow)
	{
		std::cout << "Failed to create SDL Window: " << SDL_GetError() << '\n';
		return false;
	}

	m_Camera.SetAspectRatio(float(res.Width), float(res.Height));

	m_WindowHelper.pWindow = m_pWindow;
	m_WindowHelper.pFrontBuffer = SDL_GetWindowSurface(m_pWindow);
	m_WindowHelper.pBackBuffer = SDL_CreateRGBSurface(0, res.Width, res.Height, 32, 0, 0, 0, 0);
	m_WindowHelper.pBackBufferPixels = (unsigned int*)m_WindowHelper.pBackBuffer->pixels;
	m_WindowHelper.Resolution = res;

	DisplayResolutionDetails(res);

	m_pRenderer = new CUDARenderer{ m_WindowHelper };

#ifdef BINNING
	//TODO: only works for ResolutionStandards of 4:3
	//TODO: instead of having uniform bins (5x5pixels), try having bins with dimensions that correspond with resolutionstandard (4x3/16x9pixels)
	//(Having the opposite setup atm)
	IPoint2 numBins{};
	IPoint2 binDim{};
	int binMultiplier = BINMULTIPLIER;
	int binQueueMaxSize = BINQUEUEMAXSIZE;
	if (!CheckBinMACROS(res, binMultiplier, numBins, binDim, binQueueMaxSize))
	{
		std::cout << "\n!Warning: invalid BINMULTIPLIER for current resolution!\nReverting to default value of 1\n";
	}
	m_pRenderer->SetupBins(numBins, binDim, binQueueMaxSize);
#endif

	m_pRenderer->DisplayGPUSpecs();

	return true;
}

bool Application::CheckBinMACROS(const Resolution& res, int& binMultiplier, IPoint2& numBins, IPoint2& binDim, int& binQueueMaxSize)
{
	binMultiplier = Clamp(binMultiplier, 1, 32); //results in a multiple of 2
	numBins = { (int)res.AspectRatio.w * binMultiplier, (int)res.AspectRatio.h * binMultiplier };
	binDim = { (int)res.Width / numBins.x, (int)res.Height / numBins.y };
	binQueueMaxSize = Clamp(binQueueMaxSize, 32, 1024);
	binQueueMaxSize -= binQueueMaxSize % 32; //always a multiple of 32
	const IPoint2 numThreads = { 16, 16 }; //implementation only supports 16 by 16 thread arrays
	const IPoint2 pixelCoveragePerThread = { binDim.x / numThreads.x, binDim.y / numThreads.y };
	if (pixelCoveragePerThread.x == 0 || pixelCoveragePerThread.y == 0)
	{
		binMultiplier = 1;
		numBins = { (int)res.AspectRatio.w, (int)res.AspectRatio.h };
		binDim = { (int)res.Width / numBins.x, (int)res.Height / numBins.y };
		return false;
	}
	return true;
}

void Application::PrintInfo()
{
	std::cout << "------------------------------\n";
	std::cout << "Custom CUDA Rasterizer v2.2\n";
	std::cout << "WASD to move camera\n";
	std::cout << "RMB + drag to rotate camera\n";
	std::cout << "LMB + drag to move camera along its forward vector\n";
	std::cout << "LMB & RMB to move camera upwards\n";
	std::cout << "Press R to toggle depth / normal colour visualisation\n";
	std::cout << "Press F to toggle texture sample state\n";
	std::cout << "Press C to toggle culling mode\n";
	std::cout << "------------------------------\n";
}

void Application::Run()
{
	PrintInfo();

	//Global Vars
	bool isLooping = true;
	bool takeScreenshot = false;
	float printTimer = 0.f;
	float elapsedSec = 0.f;

	//Start loop
	m_Timer.Start();
	while (isLooping)
	{
		elapsedSec = m_Timer.GetElapsed();
		//--------- Get input events ---------
		EventManager::ProcessInputs(isLooping, takeScreenshot, elapsedSec);

		//--------- Update camera ---------
		UpdateCamera(elapsedSec);

		//--------- Render ---------
#ifdef STATS_REALTIME
		m_pRenderer->StartTimer();
#endif
		m_pRenderer->RenderAuto(m_SceneGraph, m_TextureManager, m_Camera);
#ifdef STATS_REALTIME
		const unsigned int totalNumVisTris = m_pRenderer->GetTotalNumVisibleTriangles();
		const unsigned int totalNumTris = m_SceneGraph.GetTotalNumTriangles();
		const float percentageVisible = ((float)totalNumVisTris / totalNumTris) * 100.f;
		std::cout << "Visible Tri's: " << totalNumVisTris << " / " << totalNumTris << ' ';
		std::cout << '(' << percentageVisible << "%) ";
		const float ms = m_pRenderer->StopTimer();
		std::cout << "FPS: " << GetFPSImmediate(ms);
		std::cout << " (" << ms << " ms)\r";
#endif

		//--------- Timer ---------
		m_Timer.Update();

#ifndef STATS_REALTIME
		printTimer += elapsedSec;
		if (printTimer >= 1.f)
		{
			printTimer = 0.f;
#ifndef BENCHMARK
			//std::cout << "FPS: " << pTimer->GetFPS() << std::endl;
#else
			std::cout << std::endl;
#endif
		}
#endif

		//--------- Update Scene ---------
		m_SceneGraph.Update(elapsedSec);
	}
	m_Timer.Stop();
}

void Application::LoadSceneGraph(SceneGraph* pSceneGraph)
{
	m_SceneGraph.LoadSceneGraph(pSceneGraph, m_TextureManager);
}