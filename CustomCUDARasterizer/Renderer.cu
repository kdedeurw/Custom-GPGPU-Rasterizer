#include "PCH.h"
#include "Renderer.cuh"
#include <vector>

//Project includes
#include "CUDAROPs.cuh"
#include "WindowHelper.h"
#include "Mesh.h"
#include "SceneManager.h"
#include "SceneGraph.h"

GPU_KERNEL void RenderKernel(SDL_Surface* pBackBuffer, uint32_t width, uint32_t height, float* depthBuffer, Mesh::Textures* pTextures)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	//const int idx = x + y * width; //USE TO ACCESS INDIVIDUAL PIXEL IN PIXELBUFFER

	//Reimplement render pipeline from software rasterizer
	//Assign 1 triangle per thread?

	//TODO: define single triangle from mesh

	//TODO: render triangle
	//CUDAROP::RenderTriangle(triangle);

	/*
	Send 1 triangle per thread (coarse rasterizer)

	thread calculates 3 vertices, barycentric coordinates

	thread atomically does depthtest in rasterizer stage before pixelshader

	IF depthtest succeeds, shade pixel
	ELSE continue/break

	TODO: use shared memory, then coalescened copy
	e.g. single bin buffer in single shared memory

	TODO: use binning, each bin their AABBs (and checks) (bin rasterizer)
	*/
}

CPU_CALLABLE void Render(WindowHelper* pWindowHelper, Camera* pCamera)
{
	//TODO: give scenegraph to device memory!!!
	//SceneGraph
	SceneManager& sm = *SceneManager::GetInstance();
	SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pObjects = pSceneGraph->GetObjects();

	const dim3 threadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ pWindowHelper->Width / threadsPerBlock.x, pWindowHelper->Height / threadsPerBlock.y };

	//TODO: GPU Render Params, CameraData, SceneGraphData
	RenderKernel<<<numBlocks, threadsPerBlock>>>(pWindowHelper->pBackBuffer,
		pWindowHelper->Width, 
		pWindowHelper->Height, 
		pWindowHelper->pDepthBuffer,
		nullptr);

	cudaDeviceSynchronize();
}