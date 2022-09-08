#include "PCH.h"

//External includes
#include "vld.h"
#include "SDL.h"
#include "SDL_surface.h"
#undef main

//Project includes
#include "ETimer.h"
#include "EventManager.h"
#include "ObjParser.h"
#include "DirectionalLight.h"
#include "CUDATexture.h"
#include "CUDATextureManager.h"

//Project CUDA includes
#include "CUDARenderer.h"

#include <curand_kernel.h>
#include <curand.h>

void CreateScenes(SceneManager& sm, CUDATextureManager& tm)
{
	std::vector<SceneGraph*> pSceneGraphs{};
	ObjParser parser{};
	IVertex* pVertexBuffer{};
	unsigned int* pIndexBuffer{};
	unsigned int numVertices{}, numIndices{};
	short vertexType{ -1 }, vertexStride{ -1 };

	//{
	//	// SceneGraph 1
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // White small Triangle
	//		std::vector<IVertex> vertices = {
	//			{ FPoint3{ 0.f, 2.f, 0.f }, RGBColor{1.f, 1.f, 1.f} },
	//			{ FPoint3{ -1.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} },
	//			{ FPoint3{ 1.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} } };
	//		std::vector<unsigned int> indices = { 0, 1, 2 };
	//		Mesh* pTriangle = new Mesh{ vertices, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pTriangle);
	//	}
	//	{
	//		// Mesh 2 // Coloured larger Triangle
	//		std::vector<IVertex> vertices = {
	//			{ FPoint3{ 0.f, 4.f, -2.f }, RGBColor{1.f, 0.f, 0.f} },
	//			{ FPoint3{ -3.f, -2.f, -2.f }, RGBColor{0.f, 1.f, 0.f} },
	//			{ FPoint3{ 3.f, -2.f, -2.f }, RGBColor{0.f, 0.f, 1.f} } };
	//		std::vector<unsigned int> indices = { 0, 1, 2 };
	//		Mesh* pTriangle = new Mesh{ vertices, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pTriangle);
	//	}
	//	//pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	//{
	//	// SceneGraph 2
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	short vertexType{};
	//	vertexType |= (int)VertexType::Uv;
	//	const std::string uvGridTexPath = "Resources/uv_grid_2.png";
	//	CUDATexture* pUVGridTexture = new CUDATexture{ uvGridTexPath };
	//	if (!pUVGridTexture->IsAllocated())
	//	{
	//		throw std::runtime_error{ "CreateScenes > pUVGridTexture was unable to allocate" };
	//	}
	//	const int uvGridTextureId = tm.AddCUDATexture(pUVGridTexture);
	//	{
	//		numVertices = 9;
	//		numIndices = 24;
	//		// Mesh 1 // TriangleList Quad
	//		IVertex* pVertices = new IVertex[numVertices];
	//		pVertices[0] = { FPoint3{-3, 3, -2}, FVector2{0, 0} }; 
	//		pVertices[1] = { FPoint3{ 0, 3, -2 }, FVector2{ 0.5f, 0 } };
	//		pVertices[2] = { FPoint3{ 3, 3, -2 }, FVector2{ 1, 0 } };
	//		pVertices[3] = { FPoint3{ -3, 0, -2 }, FVector2{ 0, 0.5f } };
	//		pVertices[4] = { FPoint3{ 0, 0, -2 }, FVector2{ 0.5f, 0.5f } };
	//		pVertices[5] = { FPoint3{ 3, 0, -2 }, FVector2{ 1, 0.5f } };
	//		pVertices[6] = { FPoint3{ -3, -3, -2 }, FVector2{ 0, 1 } };
	//		pVertices[7] = { FPoint3{ 0, -3, -2 }, FVector2{ 0.5f, 1 } };
	//		pVertices[8] = { FPoint3{ 3, -3, -2 }, FVector2{ 1, 1 } };
	//		unsigned int* pIndices = new unsigned int[numIndices];
	//		pIndices[0] = 0; pIndices[1] = 3; pIndices[2] = 1;
	//		pIndices[3] = 3; pIndices[4] = 4; pIndices[5] = 1;
	//		pIndices[6] = 1; pIndices[7] = 4; pIndices[8] = 2;
	//		pIndices[9] = 4; pIndices[10] = 5; pIndices[11] = 2;
	//		pIndices[12] = 3; pIndices[13] = 6; pIndices[14] = 4;
	//		pIndices[15] = 6; pIndices[16] = 7; pIndices[17] = 4;
	//		pIndices[18] = 4; pIndices[19] = 7; pIndices[20] = 5;
	//		pIndices[21] = 7; pIndices[22] = 8; pIndices[23] = 5;
	//		Mesh* pTriangleListQuad = new Mesh{ pVertices, numVertices, vertexStride, vertexType, pIndices, numIndices, PrimitiveTopology::TriangleList };
	//		pTriangleListQuad->SetTextureId(uvGridTextureId, Mesh::TextureID::Diffuse);
	//		pSceneGraph->AddMesh(pTriangleListQuad);
	//	}
	//	//{
	//	//	// Mesh 2 // TriangleStrip Quad
	//	//	std::vector<IVertex> vertices = {
	//	//		{ FPoint3{-3, 3, -2}, FVector2{0, 0} }, IVertex{ FPoint3{0, 3, -2}, FVector2{0.5f, 0} }, IVertex{ FPoint3{3, 3, -2}, FVector2{1, 0} },
	//	//		{ FPoint3{-3, 0, -2}, FVector2{0, 0.5f} }, IVertex{ FPoint3{0, 0, -2}, FVector2{0.5f, 0.5f} }, IVertex{ FPoint3{3, 0, -2}, FVector2{1, 0.5f} },
	//	//		{ FPoint3{-3, -3, -2}, FVector2{0, 1} }, IVertex{ FPoint3{0, -3, -2}, FVector2{0.5f, 1} }, IVertex{ FPoint3{3, -3, -2}, FVector2{1, 1} } };
	//	//	std::vector<unsigned int> indices = { 0, 3, 1, 4, 2, 5, 5, 3, 3, 6, 4, 7, 5, 8 }; // strip
	//	//	Mesh* pTriangleStripQuad = new Mesh{ vertices, indices, PrimitiveTopology::TriangleStrip };
	//	//	pTriangleStripQuad->LoadTextures(texPaths);
	//	//	pSceneGraph->AddMesh(pTriangleStripQuad);
	//	//}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	
	//{
	//	// SceneGraph 3 // TukTuk
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // TukTuk
	//		short vertexType{};
	//		parser.OpenFile("Resources/tuktuk.obj");
	//		parser.SetInvertYAxis(true);
	//		parser.ReadFromObjFile(vertexBuffer, indexBuffer, vertexType);
	//		Mesh* pTukTukMesh = new Mesh{ vertexBuffer, indexBuffer, PrimitiveTopology::TriangleList };
	//		const std::string texPaths[4]{ "Resources/tuktuk.png", "", "", "" };
	//		pTukTukMesh->LoadTextures(texPaths);
	//		pSceneGraph->AddMesh(pTukTukMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//
	//}

	{
		// SceneGraph 4 // Bunny
		SceneGraph* pSceneGraph = new SceneGraph{};
		{
			// Mesh 1 // Bunny 
			parser.OpenFile("Resources/lowpoly_bunny.obj");
			parser.ReadFromObjFile(pVertexBuffer, numVertices, pIndexBuffer, numIndices, vertexType);
			Mesh* pBunnyMesh = new Mesh{ pVertexBuffer, numVertices, vertexStride, vertexType, pIndexBuffer, numIndices, PrimitiveTopology::TriangleList };
			pSceneGraph->AddMesh(pBunnyMesh);
		}
		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
		pSceneGraphs.push_back(pSceneGraph);
	}
	
	//{
	//	// SceneGraph 5 // Vehicle
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // Vehicle
	//		short vertexType{};
	//		parser.OpenFile("Resources/vehicle.obj");
	//		parser.SetInvertYAxis(true);
	//		parser.ReadFromObjFile(vertexBuffer, indexBuffer, vertexType);
	//		const std::string texPaths[4]{ "Resources/vehicle_diffuse.png", "Resources/vehicle_normal.png", "Resources/vehicle_specular.png", "Resources/vehicle_gloss.png" };
	//		Mesh* pVehicleMesh = new Mesh{ vertexBuffer, indexBuffer, PrimitiveTopology::TriangleList };
	//		pVehicleMesh->LoadTextures(texPaths);
	//		pSceneGraph->AddMesh(pVehicleMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	//{
	//	// SceneGraph 6 // Cube
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1
	//		std::vector<IVertex> vertices = {
	//		{ FPoint3{ 1.f, -1.f, -1.f }, RGBColor{1.f, 0.f, 0.f} },
	//		{ FPoint3{ 1.f, -1.f, 1.f }, RGBColor{0.f, 1.f, 0.f} },
	//		{ FPoint3{ -1.f, -1.f, 1.f }, RGBColor{0.f, 0.f, 1.f} },
	//		{ FPoint3{ -1.f, -1.f, -1.f }, RGBColor{1.f, 0.f, 0.f} },
	//		{ FPoint3{ 1.f, 1.f, -1.f }, RGBColor{0.f, 1.f, 0.f} },
	//		{ FPoint3{ 1.f, 1.f, 1.f }, RGBColor{0.f, 0.f, 1.f} },
	//		{ FPoint3{ -1.f, 1.f, 1.f }, RGBColor{1.f, 0.f, 0.f} },
	//		{ FPoint3{ -1.f, 1.f, -1.f }, RGBColor{0.f, 1.f, 0.f} } };
	//		std::vector<unsigned int> indices = {
	//			1, 2, 3, 7, 6, 5, 4, 5, 1, 5, 6, 2, 2, 6, 7, 0, 3, 7, 0, 1, 3, 4, 7, 5, 0, 4, 1, 1, 5, 2, 3, 2, 7, 4, 0, 7, //BackFace
	//			7, 0, 4, 7, 2, 3, 2, 5, 1, 1, 4, 0, 5, 7, 4, 3, 1, 0, 7, 3, 0, 7, 6, 2, 2, 6, 5, 1, 5, 4, 5, 6, 7, 3, 2, 1 }; //FrontFace
	//		Mesh* pCube = new Mesh{ vertices, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pCube);
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	//{
	//	// SceneGraph 7 // VertexCount
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1
	//		const unsigned int numVertices = 3 * 1000000;
	//		std::vector<IVertex> vertices{};
	//		std::vector<unsigned int> indices{};
	//		vertices.reserve(numVertices);
	//		indices.reserve(numVertices * 3);
	//		for (int i{}; i < numVertices; i += 3)
	//		{
	//			vertices.push_back({ FPoint3{ 0.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} });
	//			vertices.push_back({ FPoint3{ 0.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} });
	//			vertices.push_back({ FPoint3{ 0.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} });
	//			indices.push_back(i);
	//			indices.push_back(i + 1);
	//			indices.push_back(i + 2);
	//		}
	//		Mesh* pVertices = new Mesh{ vertices, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pVertices);
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	for (SceneGraph* pSceneGraph : pSceneGraphs)
	{
		sm.AddSceneGraph(pSceneGraph);
	}
}

void ShutDown(SDL_Window* pWindow)
{
	SDL_DestroyWindow(pWindow);
	SDL_Quit();
}

void UpdateCamera(Camera& cam, float elapsedSec)
{
	const float moveSpeed = cam.GetMoveSpeed();
	const float rotSpeed = cam.GetRotationSpeed();
	const MouseInformation mi = EventManager::GetMouseInformation();

	if (mi.scrollwheel != 0)
	{
		cam.SetMoveSpeed(moveSpeed + (float)mi.scrollwheel);
		cam.SetRotationSpeed(rotSpeed + (mi.scrollwheel / 10.f));
	}

	if (mi.lmb && mi.rmb)
	{
		cam.TranslateY(mi.y * moveSpeed * elapsedSec);
	}
	else if (mi.lmb && !mi.rmb)
	{
		cam.TranslateZ(mi.y * moveSpeed * elapsedSec);
		cam.Yaw(mi.x * rotSpeed);
	}
	else if (!mi.lmb && mi.rmb)
	{
		cam.Yaw(-mi.x * rotSpeed);
		cam.Pitch(-mi.y * rotSpeed);
	}

	if (EventManager::IsKeyDown(SDL_SCANCODE_W))
	{
		cam.TranslateZ(-moveSpeed * elapsedSec);
	}
	else if (EventManager::IsKeyDown(SDL_SCANCODE_S))
	{
		cam.TranslateZ(moveSpeed * elapsedSec);
	}
	if (EventManager::IsKeyDown(SDL_SCANCODE_A))
	{
		cam.TranslateX(-moveSpeed * elapsedSec);
	}
	else if (EventManager::IsKeyDown(SDL_SCANCODE_D))
	{
		cam.TranslateX(moveSpeed * elapsedSec);
	}
}

int GetFPSImmediate(float ms)
{
	return int(1 / ms * 1000);
}

void CUDACheckBankConflicts(unsigned int dataSizePerThread)
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

void DisplayResolutionDetails(const Resolution& res)
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

//TODO: CUDA runtime error: invalid argument
//Due to Texturing!

int main(int argc, char* args[])
{
	//Unreferenced parameters
	(void)argc;
	(void)args;

	//Single-GPU setup
	const int deviceId = 0;
	CheckErrorCuda(SetDeviceCuda(deviceId));

	//Create window + surfaces
	SDL_Init(SDL_INIT_VIDEO);

	//Select resolution
	const Resolution res = Resolution::GetResolution(Resolution::ResolutionStandard::SD);

	SDL_Window* pWindow = SDL_CreateWindow(
		"Custom GPGPU CUDA Rasterizer - GW Kristof Dedeurwaerder",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		res.Width, res.Height, 0);

	if (!pWindow)
		return 1;

	//Initialize framework
	SceneManager sm{};
	CUDATextureManager tm{};

	//Camera Setup
	//const FPoint3 camPos = { 0.f, 5.f, 65.f };
	const FPoint3 camPos = { 0.f, 1.f, 5.f };
	const float fov = 45.f;
	Camera camera{ camPos, fov };
	camera.SetAspectRatio(float(res.Width), float(res.Height));
	Elite::Timer* pTimer = new Elite::Timer();

	WindowHelper windowHelper{};
	windowHelper.pWindow = pWindow;
	windowHelper.pFrontBuffer = SDL_GetWindowSurface(pWindow);
	windowHelper.pBackBuffer = SDL_CreateRGBSurface(0, res.Width, res.Height, 32, 0, 0, 0, 0);
	windowHelper.pBackBufferPixels = (unsigned int*)windowHelper.pBackBuffer->pixels;
	windowHelper.Resolution = res;

	CreateScenes(sm, tm);

	DisplayResolutionDetails(res);

#ifdef BINNING
	const int binMultiplier = 10;
	const IPoint2 numBins = { (int)res.AspectRatio.w * binMultiplier, (int)res.AspectRatio.h * binMultiplier };
	//const IPoint2 numBins = { 40, 30 };
	const IPoint2 binDim = { (int)res.Width / numBins.x, (int)res.Height / numBins.y };
	const IPoint2 numThreads = { 16, 16 };
	const IPoint2 pixelCoveragePerThread = { binDim.y / numThreads.y, binDim.y / numThreads.y };
	const int binQueueMaxSize = 256;
	CUDARenderer* pCudaRenderer = new CUDARenderer{ windowHelper, numBins, binDim, binQueueMaxSize };
#else
	CUDARenderer* pCudaRenderer = new CUDARenderer{ windowHelper };
#endif
	SceneGraph* pSceneGraph = sm.GetSceneGraph();
	pCudaRenderer->LoadScene(pSceneGraph, tm);
	pCudaRenderer->DisplayGPUSpecs();

	std::cout << "------------------------------\n";
	std::cout << "Custom CUDA Rasterizer v2.0\n";
	std::cout << "WASD to move camera\n";
	std::cout << "RMB + drag to rotate camera\n";
	std::cout << "LMB + drag to move camera along its forward vector\n";
	std::cout << "LMB & RMB to move camera upwards\n";
	std::cout << "Press R to toggle depth colour visualisation\n";
	std::cout << "Press F to toggle texture sample state\n";
	std::cout << "Press C to toggle culling mode\n";
	std::cout << "------------------------------\n";

	//Global Vars
	bool isLooping = true;
	bool takeScreenshot = false;
	float printTimer = 0.f;
	float elapsedSec = 0.f;

	//Start loop
	pTimer->Start();
	while (isLooping)
	{
		elapsedSec = pTimer->GetElapsed();
		//--------- Get input events ---------
		EventManager::ProcessInputs(isLooping, takeScreenshot, elapsedSec);

		//--------- Update camera ---------
		UpdateCamera(camera, elapsedSec);

		//--------- Render ---------
#ifdef STATS_REALTIME
		pCudaRenderer->StartTimer();
#endif
		pCudaRenderer->RenderAuto(sm, tm, &camera);
#ifdef STATS_REALTIME
		const unsigned int totalNumVisTris = pCudaRenderer->GetTotalNumVisibleTriangles();
		const unsigned int totalNumTris = pCudaRenderer->GetTotalNumTriangles();
		const float percentageVisible = ((float)totalNumVisTris / totalNumTris) * 100.f;
		std::cout << "Visible Tri's: " << totalNumVisTris << " / " << totalNumTris << ' ';
		std::cout << '(' << percentageVisible << "%) ";
		const float ms = pCudaRenderer->StopTimer();
		std::cout << "FPS: " << GetFPSImmediate(ms);
		std::cout << " (" << ms << " ms)\r";
#endif

		//--------- Timer ---------
		pTimer->Update();

#ifndef STATS_REALTIME
		printTimer += elapsedSec;
		if (printTimer >= 1.f)
		{
			printTimer = 0.f;
#ifndef BENCHMARK
			std::cout << "FPS: " << pTimer->GetFPS() << std::endl;
#else
			std::cout << std::endl;
#endif
		}
#endif

		//--------- Update Scenes ---------
		sm.Update(elapsedSec);
	}
	pTimer->Stop();

	//Shutdown framework
	CheckErrorCuda(DeviceSynchroniseCuda());
	if (pCudaRenderer)
		delete pCudaRenderer;
	CheckErrorCuda(DeviceResetCuda());

	if (pTimer)
		delete pTimer;

	ShutDown(pWindow);
	return 0;
}