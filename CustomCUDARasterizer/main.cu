#include "PCH.h"

//External includes
#include "vld.h"
#include "SDL.h"
#include "SDL_surface.h"
#undef main

//Standard includes
#include <iostream>
#include <algorithm>

//Project includes
#include "ETimer.h"
#include "ERenderer.h"
#include "Math.h"
#include "MathUtilities.h"
#include "RGBColor.h"
#include "Camera.h"
#include "EventManager.h"
#include "Vertex.h"
#include "Mesh.h"
#include "Texture.h"
#include "ObjParser.h"
#include "SceneManager.h"
#include "SceneGraph.h"
#include "DirectionalLight.h"
#include "WindowHelper.h"
#include "PrimitiveTopology.h"

//Project CUDA includes
#include "CUDARenderer.cuh"

void CreateScenes(SceneManager& sm)
{
	ObjParser parser{};
	std::vector<IVertex> vertexBuffer{};
	std::vector<unsigned int> indexBuffer{};
	std::vector<SceneGraph*> pSceneGraphs{};

	{
		// SceneGraph 1
		SceneGraph* pSceneGraph = new SceneGraph{};
		//{
		//	// Mesh 1 // Triangle 1
		//	std::vector<IVertex> vertices{
		//		IVertex{ FPoint3{ 0.f, 2.f, 0.f }, FVector2{} },
		//		IVertex{ FPoint3{ -1.f, 0.f, 0.f }, FVector2{}},
		//		IVertex{ FPoint3{ 1.f, 0.f, 0.f }, FVector2{} } };
		//	std::vector<unsigned int> indices{ 0, 1, 2 };
		//	const std::string texPaths[4]{ "", "", "", "" };
		//	Mesh* pTriangle = new Mesh{ vertices, indices, texPaths, PrimitiveTopology::TriangleList };
		//	pSceneGraph->AddMesh(pTriangle);
		//}
		{
			// Mesh 2 // Triangle 2
			std::vector<IVertex> vertices{
				IVertex{ FPoint3{ 0.f, 4.f, -2.f }, FVector2{}, FVector3{1.f,1.f,1.f}, RGBColor{1.f, 0.f, 0.f} },
				IVertex{ FPoint3{ -3.f, -2.f, -2.f }, FVector2{}, FVector3{1.f,1.f,1.f}, RGBColor{0.f, 1.f, 0.f} },
				IVertex{ FPoint3{ 3.f, -2.f, -2.f }, FVector2{}, FVector3{1.f,1.f,1.f}, RGBColor{0.f, 0.f, 1.f} } };
			std::vector<unsigned int> indices{ 0, 1, 2 };
			const std::string texPaths[4]{ "", "", "", "" };
			Mesh* pTriangle = new Mesh{ vertices, indices, texPaths, PrimitiveTopology::TriangleList };
			pSceneGraph->AddMesh(pTriangle);
		}
		//pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
		pSceneGraphs.push_back(pSceneGraph);
	}

	//{
	//	// SceneGraph 2
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	std::vector<IVertex> vertices{
	//		IVertex{FPoint3{-3, 3, -2}, FVector2{0, 0}}, IVertex{FPoint3{0, 3, -2}, FVector2{0.5f, 0}}, IVertex{FPoint3{3, 3, -2}, FVector2{1, 0}},
	//		IVertex{FPoint3{-3, 0, -2}, FVector2{0, 0.5f}}, IVertex{FPoint3{0, 0, -2}, FVector2{0.5f, 0.5f}}, IVertex{FPoint3{3, 0, -2}, FVector2{1, 0.5f}},
	//		IVertex{FPoint3{-3, -3, -2}, FVector2{0, 1}}, IVertex{FPoint3{0, -3, -2}, FVector2{0.5f, 1}}, IVertex{FPoint3{3, -3, -2}, FVector2{1, 1}} };
	//	// shared vertices among both quads (duh they're the same quad)
	//	const std::string texPaths[4]{ "Resources/uv_grid_2.png", "", "", "" };
	//	//{
	//	//	// Mesh 1 // TriangleList Quad
	//	//	std::vector<unsigned int> indices{ 0, 3, 1,
	//	//								3, 4, 1,
	//	//								1, 4, 2,
	//	//								4, 5, 2,
	//	//								3, 6, 4,
	//	//								6, 7, 4,
	//	//								4, 7, 5,
	//	//								7, 8, 5, }; // obviously a list
	//	//	Mesh* pTriangleListQuad = new Mesh{ vertices, indices, texPaths, PrimitiveTopology::TriangleList };
	//	//	pSceneGraph->AddMesh(pTriangleListQuad);
	//	//}
	//	{
	//		// Mesh 2 // TriangleStrip Quad
	//		std::vector<unsigned int> indices{ 0, 3, 1, 4, 2, 5, 5, 3, 3, 6, 4, 7, 5, 8 }; // strip
	//		Mesh* pTriangleStripQuad = new Mesh{ vertices, indices, texPaths, PrimitiveTopology::TriangleStrip };
	//		pSceneGraph->AddMesh(pTriangleStripQuad);
	//		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	//
	//{
	//	// SceneGraph 3 // TukTuk
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // TukTuk
	//		parser.OpenFile("Resources/tuktuk.obj");
	//		parser.SetInvertYAxis(true);
	//		parser.ReadFromObjFile(vertexBuffer, indexBuffer);
	//		const std::string texPaths[4]{ "Resources/tuktuk.png", "", "", "" };
	//		Mesh* pTukTukMesh = new Mesh{ vertexBuffer, indexBuffer, texPaths, PrimitiveTopology::TriangleList, 1.f };
	//		pSceneGraph->AddMesh(pTukTukMesh);
	//		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	//
	//{
	//	// SceneGraph 4 // Bunny
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // Bunny 
	//		parser.OpenFile("Resources/lowpoly_bunny.obj");
	//		parser.ReadFromObjFile(vertexBuffer, indexBuffer);
	//		const std::string texPaths[4]{ "", "", "", "" };
	//		Mesh* pBunnyMesh = new Mesh{ vertexBuffer, indexBuffer, texPaths, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pBunnyMesh);
	//		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	//
	//{
	//	// SceneGraph 5 // Vehicle
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // Vehicle
	//		parser.OpenFile("Resources/vehicle.obj");
	//		parser.SetInvertYAxis(true);
	//		parser.ReadFromObjFile(vertexBuffer, indexBuffer);
	//		const std::string texPaths[4]{ "Resources/vehicle_diffuse.png", "Resources/vehicle_normal.png", "Resources/vehicle_specular.png", "Resources/vehicle_gloss.png" };
	//		Mesh* pVehicleMesh = new Mesh{ vertexBuffer, indexBuffer, texPaths, PrimitiveTopology::TriangleList, 1.f };
	//		pSceneGraph->AddMesh(pVehicleMesh);
	//		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
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

int main(int argc, char* args[])
{
	//Single-GPU setup
	const int deviceId = 0;
	CheckErrorCuda(SetDeviceCuda(deviceId));

	//Unreferenced parameters
	(void)argc;
	(void)args;

	//Create window + surfaces
	SDL_Init(SDL_INIT_VIDEO);

	const uint32_t width = 640;
	const uint32_t height = 480;
	SDL_Window* pWindow = SDL_CreateWindow(
		"Custom GPGPU CUDA Rasterizer - GW Kristof Dedeurwaerder",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		width, height, 0);

	if (!pWindow)
		return 1;

	//Initialize framework
	Elite::Timer* pTimer = new Elite::Timer();
	Elite::Renderer* pRenderer = new Elite::Renderer(pWindow);

	SceneManager sm{};

	Camera camera{ FPoint3{ 0.f, 0.f, 10.f }, 45.f };
	camera.SetAspectRatio(float(width), float(height));
	pRenderer->SetCamera(&camera);

	WindowHelper windowHelper{};
	windowHelper.pWindow = pWindow;
	windowHelper.Width = width;
	windowHelper.Height = height;
	windowHelper.pFrontBuffer = SDL_GetWindowSurface(pWindow);
	windowHelper.pBackBuffer = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
	windowHelper.pBackBufferPixels = (unsigned int*)windowHelper.pBackBuffer->pixels;

	CreateScenes(sm);

	CUDARenderer* pCudaRenderer = new CUDARenderer{ windowHelper };
	pCudaRenderer->LoadScene(sm.GetSceneGraph());

	//Start loop
	pTimer->Start();
	bool isLooping = true;
	bool takeScreenshot = false;
	float printTimer = 0.f;
	float elapsedSec{};
	while (isLooping)
	{
		elapsedSec = pTimer->GetElapsed();
		//--------- Get input events ---------
		EventManager::ProcessInputs(isLooping, takeScreenshot, elapsedSec);

		//--------- Update camera ---------
		camera.Update(elapsedSec);

		//--------- Render ---------
		//pRenderer->Render(sm);
		//pCudaRenderer->StartTimer();
		pCudaRenderer->Render(sm, &camera);
		//const float ms = pCudaRenderer->StopTimer();
		//std::cout << "CUDARenderer total rendering time: " << ms << "ms\n";
		//std::cout << "FPS: " << 1 / ms * 1000  << '\n';

		//--------- Timer ---------
		pTimer->Update();
		printTimer += elapsedSec;
		if (printTimer >= 1.f)
		{
			printTimer = 0.f;
			std::cout << "FPS: " << pTimer->GetFPS() << std::endl;
		}

		//--------- Update Meshes ---------
		//sm.Update(elapsedSec);

		//Save screenshot after full render
		//if (takeScreenshot)
		//{
		//	if (!pRenderer->SaveBackbufferToImage())
		//		std::cout << "Screenshot saved!" << std::endl;
		//	else
		//		std::cout << "Something went wrong. Screenshot not saved!" << std::endl;
		//	takeScreenshot = false;
		//}
	}
	pTimer->Stop();

	CheckErrorCuda(DeviceSynchroniseCuda());
	CheckErrorCuda(DeviceResetCuda());

	//Shutdown framework
	delete pRenderer;
	delete pTimer;
	delete pCudaRenderer;

	ShutDown(pWindow);
	return 0;
}