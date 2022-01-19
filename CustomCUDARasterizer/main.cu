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

#include "Renderer.cuh"
#include "WindowHelper.h"

//Choose which GPU to run on, change this on a multi-GPU system. (Default is 0, for single-GPU systems)
cudaError_t SetDeviceCuda(int deviceId = 0);
//Calls cudaGetLastError, this checks for any errors launching the kernel
cudaError_t CheckErrorCuda();
//Calls cudaDeviceSynchronize, this waits for the kernel to finish, and returns any errors encountered during the launch.
cudaError_t DeviceSynchroniseCuda();
//Calls cudaDeviceReset, this must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
cudaError_t DeviceResetCuda();

void Init()
{
	std::vector<SceneGraph*> pSceneGraphs{};

	//{
	//	// SceneGraph 1
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1
	//		std::vector<IVertex> vertices{
	//			IVertex{ FPoint3{ 0.f, 2.f, 0.f }, FVector2{} },
	//			IVertex{ FPoint3{ -1.f, 0.f, 0.f }, FVector2{}},
	//			IVertex{ FPoint3{ 1.f, 0.f, 0.f }, FVector2{} } };
	//		std::vector<int> indices{ 0, 1, 2 };
	//		const std::string texPaths[4]{ "", "", "", "" };
	//		Mesh* pMesh = new Mesh{ vertices, indices, texPaths, Mesh::PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pMesh);
	//	}
	//	{
	//		// Mesh 2
	//		std::vector<IVertex> vertices{
	//			IVertex{ FPoint3{ 0.f, 4.f, -2.f }, FVector2{}, FVector3{1.f,1.f,1.f}, RGBColor{1.f, 0.f, 0.f} },
	//			IVertex{ FPoint3{ -3.f, -2.f, -2.f }, FVector2{}, FVector3{1.f,1.f,1.f}, RGBColor{0.f, 1.f, 0.f} },
	//			IVertex{ FPoint3{ 3.f, -2.f, -2.f }, FVector2{}, FVector3{1.f,1.f,1.f}, RGBColor{0.f, 0.f, 1.f} } };
	//		std::vector<int> indices{ 0, 1, 2 };
	//		const std::string texPaths[4]{ "", "", "", "" };
	//		Mesh* pMesh = new Mesh{ vertices, indices, texPaths, Mesh::PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	//{
	//	// SceneGraph 2
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	std::vector<IVertex> vertices{
	//		IVertex{FPoint3{-3, 3, -2}, FVector2{0, 0}}, IVertex{FPoint3{0, 3, -2}, FVector2{0.5f, 0}}, IVertex{FPoint3{3, 3, -2}, FVector2{1, 0}},
	//		IVertex{FPoint3{-3, 0, -2}, FVector2{0, 0.5f}}, IVertex{FPoint3{0, 0, -2}, FVector2{0.5f, 0.5f}}, IVertex{FPoint3{3, 0, -2}, FVector2{1, 0.5f}},
	//		IVertex{FPoint3{-3, -3, -2}, FVector2{0, 1}}, IVertex{FPoint3{0, -3, -2}, FVector2{0.5f, 1}}, IVertex{FPoint3{3, -3, -2}, FVector2{1, 1}} };
	//	// shared vertices among both quads (duh they're the same quad)
	//	const std::string texPaths[4]{ "Resources/uv_grid_2.png", "", "", "" };
	//	{
	//		// Mesh 1
	//		std::vector<int> indices{ 0, 3, 1,
	//									3, 4, 1,
	//									1, 4, 2,
	//									4, 5, 2,
	//									3, 6, 4,
	//									6, 7, 4,
	//									4, 7, 5,
	//									7, 8, 5, }; // obviously a list
	//		Mesh* pMesh = new Mesh{ vertices, indices, texPaths, Mesh::PrimitiveTopology::TriangleList };
	//		//pSceneGraph->AddMesh(pMesh);
	//	}
	//	{
	//		// Mesh 2
	//		std::vector<int> indices{ 0, 3, 1, 4, 2, 5, 5, 3, 3, 6, 4, 7, 5, 8 }; // strip
	//		Mesh* pMesh = new Mesh{ vertices, indices, texPaths, Mesh::PrimitiveTopology::TriangleStrip };
	//		pSceneGraph->AddMesh(pMesh);
	//		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	
	//{
	//	// SceneGraph 3 // TukTuk
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1
	//		ObjParser parser{ "Resources/tuktuk.obj" };
	//		parser.SetInvertYAxis(true);
	//		parser.ReadFromObjFile();
	//		std::vector<IVertex> vertices{ *parser.GetVertexBuffer() };
	//		std::vector<int> indices{ parser.GetIndexBuffer() };
	//		const std::string texPaths[4]{ "Resources/tuktuk.png", "", "", "" };
	//		Mesh* pMesh = new Mesh{ vertices, indices, texPaths, Mesh::PrimitiveTopology::TriangleList, 1.f };
	//		pSceneGraph->AddMesh(pMesh);
	//		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	
	//{
	//	// SceneGraph 4 // Bunny
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1
	//		ObjParser parser{ "Resources/lowpoly_bunny.obj" };
	//		parser.ReadFromObjFile();
	//		std::vector<IVertex> vertices{ *parser.GetVertexBuffer() };
	//		std::vector<int> indices{ parser.GetIndexBuffer() };
	//		const std::string texPaths[4]{ "", "", "", "" };
	//		Mesh* pMesh = new Mesh{ vertices, indices, texPaths, Mesh::PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pMesh);
	//		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	
	{
		// SceneGraph 5 // Vehicle
		SceneGraph* pSceneGraph = new SceneGraph{};
		{
			// Mesh 1
			ObjParser parser{ "Resources/vehicle.obj" };
			parser.SetInvertYAxis(true);
			parser.ReadFromObjFile();
			std::vector<IVertex> vertices{ *parser.GetVertexBuffer() };
			std::vector<int> indices{ parser.GetIndexBuffer() };
			const std::string texPaths[4]{ "Resources/vehicle_diffuse.png", "Resources/vehicle_normal.png", "Resources/vehicle_specular.png", "Resources/vehicle_gloss.png" };
			Mesh* pMesh = new Mesh{ vertices, indices, texPaths, Mesh::PrimitiveTopology::TriangleList, 1.f };
			pSceneGraph->AddMesh(pMesh);
			pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
		}
		pSceneGraphs.push_back(pSceneGraph);
	}

	SceneManager& sm = *SceneManager::GetInstance();
	for (SceneGraph* pSceneGraph : pSceneGraphs)
	{
		sm.AddSceneGraph(pSceneGraph);
	}
}

void Test()
{
	//size_t size;
	//size = mesh.vertexCount * sizeof(vertex_t);
	//if (cudaMalloc((void**)&mesh.d_vertices, size) == cudaErrorMemoryAllocation)
	//	printf("error creating memory for vertices\n");
	//cudaMemcpy(mesh.d_vertices, mesh.vertices, size, cudaMemcpyHostToDevice);
	//free(mesh.vertices);
	//
	//size = mesh.triangleCount * sizeof(ivec3_t);
	//if (cudaMalloc((void**)&mesh.d_triangles, size) == cudaErrorMemoryAllocation)
	//	printf("error creating memory for triangles\n");
	//cudaMemcpy(mesh.d_triangles, mesh.triangles, size, cudaMemcpyHostToDevice);
	//free(mesh.triangles);
	//
	//// allocate the polygons
	//mesh.polygonCount = mesh.triangleCount;
	//size = mesh.polygonCount * sizeof(polygon_t);
	//if (cudaMalloc((void**)&mesh.d_polygons, size) == cudaErrorMemoryAllocation)
	//	printf("error creating memory for polygons\n");
	//
	//// ----------------
	//
	//// create a color buffer on the device
	//size = width * height * sizeof(int);
	////printf("size: %d\nwidth * height: %d\nsize of color_t: %d\n", size, width * height, sizeof(color_t));
	//if (cudaMalloc((void**)&buffers.d_colorBuffer, size) == cudaErrorMemoryAllocation)
	//	printf("error creating color buffer\n");
	//
	//// create a depth buffer on the device
	//size = width * height * sizeof(float);
	//if (cudaMalloc((void**)&buffers.d_zBuffer, size) == cudaErrorMemoryAllocation)
	//	printf("error creating depth buffer\n");
	//
	//// create a lock buffer on the device
	//size = width * height * sizeof(int);
	//if (cudaMalloc((void**)&buffers.d_locks, size) == cudaErrorMemoryAllocation)
	//	printf("error creating lock buffer\n");
	//
	//// ------------------
	//
	//// copy the color buffer to host
	//size = width * height * sizeof(int);
	//buffers.colorBuffer = (color_t*)malloc(size);
	//cudaMemcpy(buffers.colorBuffer, buffers.d_colorBuffer, size, cudaMemcpyDeviceToHost);
	//
	//// free the buffers on the device
	//cudaFree(buffers.d_colorBuffer);
	//cudaFree(buffers.d_zBuffer);
	//
	//// free the polygons, vertices, and triangles on the device
	//cudaFree(mesh.d_polygons);
	//cudaFree(mesh.d_vertices);
	//cudaFree(mesh.d_triangles);
}

void ShutDown(SDL_Window* pWindow)
{
	SDL_DestroyWindow(pWindow);
	SDL_Quit();
}

int main(int argc, char* args[])
{
    cudaError_t cudaStatus{};
    cudaStatus = SetDeviceCuda();

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

	//Initialize "framework"
	Elite::Timer* pTimer = new Elite::Timer();
	Elite::Renderer* pRenderer = new Elite::Renderer(pWindow);

	Camera camera{ FPoint3{ 0.f, 5.f, 65.f }, 45.f };
	camera.SetAspectRatio(float(width), float(height));
	pRenderer->SetCamera(&camera);
	SceneManager& sm = *SceneManager::GetInstance();

	WindowHelper windowHelper{};
	windowHelper.pWindow = pWindow;
	windowHelper.Width = width;
	windowHelper.Height = height;
	windowHelper.pFrontBuffer = SDL_GetWindowSurface(pWindow);
	windowHelper.pBackBuffer = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
	windowHelper.pBackBufferPixels = (uint32_t*)windowHelper.pBackBuffer->pixels;
	float* pDepthBuffer = new float[width * height];
	std::memset(pDepthBuffer, INT_MAX, sizeof(float) * width * height);
	windowHelper.pDepthBuffer = pDepthBuffer;

	Init();

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
		pRenderer->Render();
		//Render(&windowHelper, &camera);

		//--------- Timer ---------
		pTimer->Update();
		printTimer += elapsedSec;
		if (printTimer >= 1.f)
		{
			printTimer = 0.f;
			std::cout << "FPS: " << pTimer->GetFPS() << std::endl;
		}

		//--------- Update Meshes ---------
		sm.Update(elapsedSec);

		//Save screenshot after full render
		if (takeScreenshot)
		{
			if (!pRenderer->SaveBackbufferToImage())
				std::cout << "Screenshot saved!" << std::endl;
			else
				std::cout << "Something went wrong. Screenshot not saved!" << std::endl;
			takeScreenshot = false;
		}
	}
	pTimer->Stop();

	DeviceSynchroniseCuda();

	//Shutdown "framework"
	delete pRenderer;
	delete pTimer;

	delete[] pDepthBuffer;

	// delete singleton objects
	delete SceneManager::GetInstance();

	ShutDown(pWindow);
	return 0;
}

cudaError_t SetDeviceCuda(int deviceId)
{
    cudaError_t cudaStatus = cudaSetDevice(deviceId);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    return cudaStatus;
}

cudaError_t CheckErrorCuda()
{
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "latest kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    return cudaStatus;
}

cudaError_t DeviceSynchroniseCuda()
{
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
    }
    return cudaStatus;
}

cudaError_t DeviceResetCuda()
{
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    return cudaStatus;
}