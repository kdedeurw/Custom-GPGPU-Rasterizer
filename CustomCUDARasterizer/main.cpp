#include "PCH.h"

//External includes
#include "vld.h"
#include "SDL.h"
#include "SDL_surface.h"
#undef main

#include "DEFINES.h"

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
#include "PrimitiveTopology.h"
#include "ObjParser.h"
#include "SceneManager.h"
#include "SceneGraph.h"
#include "DirectionalLight.h"
#include "WindowHelper.h"
#include "GPUTextures.h"

//Project CUDA includes
#include "CUDARenderer.cuh"

void CreateScenes(SceneManager& sm)
{
	std::vector<SceneGraph*> pSceneGraphs{};
	ObjParser parser{};
	std::vector<IVertex> vertexBuffer{};
	std::vector<unsigned int> indexBuffer{};

	//{
	//	// SceneGraph 1
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // Triangle 1
	//		short vertexType{};
	//		vertexType |= (int)VertexType::Col;
	//		std::vector<IVertex> vertices{
	//			IVertex{ FPoint3{ 0.f, 2.f, 0.f }, RGBColor{1.f, 1.f, 1.f} },
	//			IVertex{ FPoint3{ -1.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} },
	//			IVertex{ FPoint3{ 1.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} } };
	//		std::vector<unsigned int> indices{ 0, 1, 2 };
	//		Mesh* pTriangle = new Mesh{ vertices, indices, vertexType, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pTriangle);
	//	}
	//	{
	//		// Mesh 2 // Triangle 2
	//		short vertexType{};
	//		vertexType |= (int)VertexType::Col;
	//		std::vector<IVertex> vertices{
	//			IVertex{ FPoint3{ 0.f, 4.f, -2.f }, RGBColor{1.f, 0.f, 0.f} },
	//			IVertex{ FPoint3{ -3.f, -2.f, -2.f }, RGBColor{0.f, 1.f, 0.f} },
	//			IVertex{ FPoint3{ 3.f, -2.f, -2.f }, RGBColor{0.f, 0.f, 1.f} } };
	//		std::vector<unsigned int> indices{ 0, 1, 2 };
	//		Mesh* pTriangle = new Mesh{ vertices, indices, vertexType, PrimitiveTopology::TriangleList };
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
	//	std::vector<IVertex> vertices{
	//		IVertex{FPoint3{-3, 3, -2}, FVector2{0, 0}}, IVertex{FPoint3{0, 3, -2}, FVector2{0.5f, 0}}, IVertex{FPoint3{3, 3, -2}, FVector2{1, 0}},
	//		IVertex{FPoint3{-3, 0, -2}, FVector2{0, 0.5f}}, IVertex{FPoint3{0, 0, -2}, FVector2{0.5f, 0.5f}}, IVertex{FPoint3{3, 0, -2}, FVector2{1, 0.5f}},
	//		IVertex{FPoint3{-3, -3, -2}, FVector2{0, 1}}, IVertex{FPoint3{0, -3, -2}, FVector2{0.5f, 1}}, IVertex{FPoint3{3, -3, -2}, FVector2{1, 1}} };
	//	const std::string texPaths[4]{ "Resources/uv_grid_2.png", "", "", "" };
	//	{
	//		// Mesh 1 // TriangleList Quad
	//		std::vector<unsigned int> indices{ 0, 3, 1,
	//									3, 4, 1,
	//									1, 4, 2,
	//									4, 5, 2,
	//									3, 6, 4,
	//									6, 7, 4,
	//									4, 7, 5,
	//									7, 8, 5, }; // obviously a list
	//		Mesh* pTriangleListQuad = new Mesh{ vertices, indices, vertexType, PrimitiveTopology::TriangleList };
	//		pTriangleListQuad->LoadTextures(texPaths);
	//		pSceneGraph->AddMesh(pTriangleListQuad);
	//	}
	//	//{
	//	//	// Mesh 2 // TriangleStrip Quad
	//	//	std::vector<unsigned int> indices{ 0, 3, 1, 4, 2, 5, 5, 3, 3, 6, 4, 7, 5, 8 }; // strip
	//	//	Mesh* pTriangleStripQuad = new Mesh{ vertices, indices, vertexType, PrimitiveTopology::TriangleStrip };
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
	//		const std::string texPaths[4]{ "Resources/tuktuk.png", "", "", "" };
	//		Mesh* pTukTukMesh = new Mesh{ vertexBuffer, indexBuffer, vertexType, PrimitiveTopology::TriangleList };
	//		pTukTukMesh->LoadTextures(texPaths);
	//		pSceneGraph->AddMesh(pTukTukMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//
	//}

	//{
	//	// SceneGraph 4 // Bunny
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // Bunny 
	//		short vertexType{};
	//		parser.OpenFile("Resources/lowpoly_bunny.obj");
	//		parser.ReadFromObjFile(vertexBuffer, indexBuffer, vertexType);
	//		Mesh* pBunnyMesh = new Mesh{ vertexBuffer, indexBuffer, vertexType, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pBunnyMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	
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
	//		Mesh* pVehicleMesh = new Mesh{ vertexBuffer, indexBuffer, vertexType, PrimitiveTopology::TriangleList };
	//		pVehicleMesh->LoadTextures(texPaths);
	//		pSceneGraph->AddMesh(pVehicleMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	{
		// SceneGraph 6 // Cube
		SceneGraph* pSceneGraph = new SceneGraph{};
		{
			// Mesh 1
			short vertexType{};
			vertexType |= (int)VertexType::Col;
			std::vector<IVertex> vertices{
				IVertex{ FPoint3{ 1.f, -1.f, -1.f }, RGBColor{1.f, 0.f, 0.f} },
				IVertex{ FPoint3{ 1.f, -1.f, 1.f }, RGBColor{0.f, 1.f, 0.f} },
				IVertex{ FPoint3{ -1.f, -1.f, 1.f }, RGBColor{0.f, 0.f, 1.f} },
				IVertex{ FPoint3{ -1.f, -1.f, -1.f }, RGBColor{1.f, 0.f, 0.f} },
				IVertex{ FPoint3{ 1.f, 1.f, -1.f }, RGBColor{0.f, 1.f, 0.f} },
				IVertex{ FPoint3{ 1.f, 1.f, 1.f }, RGBColor{0.f, 0.f, 1.f} },
				IVertex{ FPoint3{ -1.f, 1.f, 1.f }, RGBColor{1.f, 0.f, 0.f} },
				IVertex{ FPoint3{ -1.f, 1.f, -1.f }, RGBColor{0.f, 1.f, 0.f} } };
			std::vector<unsigned int> indices{ 
				1, 2, 3, 7, 6, 5, 4, 5, 1, 5, 6, 2, 2, 6, 7, 0, 3, 7, 0, 1, 3, 4, 7, 5, 0, 4, 1, 1, 5, 2, 3, 2, 7, 4, 0, 7, //BackFace
				7, 0, 4, 7, 2, 3, 2, 5, 1, 1, 4, 0, 5, 7, 4, 3, 1, 0, 7, 3, 0, 7, 6, 2, 2, 6, 5, 1, 5, 4, 5, 6, 7, 3, 2, 1, //FrontFace
				};
			Mesh* pCube = new Mesh{ vertices, indices, vertexType, PrimitiveTopology::TriangleList };
			pSceneGraph->AddMesh(pCube);
		}
		pSceneGraphs.push_back(pSceneGraph);
	}

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

int GetFPSImmediate(float ms)
{
	return int(1 / ms * 1000);
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
	SceneManager sm{};
	//Camera camera{ FPoint3{ 0.f, 5.f, 65.f }, 45.f };
	Camera camera{ FPoint3{ 0.f, 0.f, 5.f }, 45.f };
	camera.SetAspectRatio(float(width), float(height));
	Elite::Timer* pTimer = new Elite::Timer();

	WindowHelper windowHelper{};
	windowHelper.pWindow = pWindow;
	windowHelper.Width = width;
	windowHelper.Height = height;
	windowHelper.pFrontBuffer = SDL_GetWindowSurface(pWindow);
	windowHelper.pBackBuffer = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
	windowHelper.pBackBufferPixels = (unsigned int*)windowHelper.pBackBuffer->pixels;

	CreateScenes(sm);

#ifdef HARDWARE_ACCELERATION
	CUDARenderer* pCudaRenderer = new CUDARenderer{ windowHelper };
	pCudaRenderer->LoadScene(sm.GetSceneGraph());
#else
	Elite::Renderer* pRenderer = new Elite::Renderer(pWindow);
	pRenderer->SetCamera(&camera);
#endif

	std::cout << "------------------------------\n";
	std::cout << "Custom CUDA Rasterizer v1.0\n";
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
		camera.Update(elapsedSec);

		//--------- Render ---------
#ifdef HARDWARE_ACCELERATION
#ifdef FPS_REALTIME
		pCudaRenderer->StartTimer();
#endif
		pCudaRenderer->RenderAuto(sm, &camera);
#ifdef FPS_REALTIME
		const float ms = pCudaRenderer->StopTimer();
		std::cout << "FPS: " << GetFPSImmediate(ms);
		std::cout << " (" << ms << " ms)\r";
#endif
#else
		pRenderer->Render(sm);
#endif

		//--------- Timer ---------
		pTimer->Update();
#ifndef FPS_REALTIME
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

		//--------- Update Meshes ---------
		sm.Update(elapsedSec);

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

	//Shutdown framework
#ifdef HARDWARE_ACCELERATION
	CheckErrorCuda(DeviceSynchroniseCuda());
	delete pCudaRenderer;
	CheckErrorCuda(DeviceResetCuda());
#else
	delete pRenderer;
#endif
	delete pTimer;


	ShutDown(pWindow);
	return 0;
}