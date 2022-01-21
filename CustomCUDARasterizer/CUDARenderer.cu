#include "PCH.h"
#include "CUDARenderer.cuh"
#include <vector>

//Project CUDA includes
#include "CUDAROPs.cuh"
#include "GPUTextureSampler.cuh"

//Project includes
#include "WindowHelper.h"
#include "SceneManager.h"
#include "SceneGraph.h"
#include "Mesh.h"
#include "Camera.h"
#include "Vertex.h"
#include "BoundingBox.h"
#include "GPUTextures.h"

CPU_CALLABLE CUDARenderer::CUDARenderer(const WindowHelper& windowHelper)
	: m_WindowHelper{ windowHelper }
	, m_dev_IVertexBuffer{}
	, m_dev_IndexBuffer{}
	, m_dev_OVertexBuffer{}
	, m_dev_PixelShaderBuffer{}
	, m_dev_FrameBuffer{}
	, m_dev_DepthBuffer{}
	, m_dev_RenderData{}
	, m_NumThreadsPerBlock{ 16, 16 }
	, m_NumBlocks{}
{
	m_NumBlocks = { m_WindowHelper.Width / m_NumThreadsPerBlock.x, m_WindowHelper.Height / m_NumThreadsPerBlock.y };
	InitCUDARasterizer();
}

CPU_CALLABLE CUDARenderer::~CUDARenderer()
{
	FreeCUDARasterizer();
}

CPU_CALLABLE int CUDARenderer::SwapBuffers()
{
	int state = SDL_LockSurface(m_WindowHelper.pBackBuffer);
	SDL_UnlockSurface(m_WindowHelper.pBackBuffer);
	SDL_BlitSurface(m_WindowHelper.pBackBuffer, 0, m_WindowHelper.pFrontBuffer, 0);
	return state;
}

CPU_CALLABLE int CUDARenderer::Present()
{
	//Present new frame
	int state = SDL_UpdateWindowSurface(m_WindowHelper.pWindow);
	return state;
}

CPU_CALLABLE void CUDARenderer::Render(const SceneManager& sm, const Camera* pCamera)
{
	//Clear screen and reset buffers
	Clear();

	//Render Data
	const bool isDepthColour = sm.IsDepthColour();
	const SampleState sampleState = sm.GetSampleState();

	//Camera Data
	const FPoint3 camPos = pCamera->GetPos();
	const FMatrix4 lookatMatrix = pCamera->GetLookAtMatrix();
	const FMatrix4 viewMatrix = pCamera->GetViewMatrix(lookatMatrix);
	const FMatrix4 projectionMatrix = pCamera->GetProjectionMatrix();
	const FMatrix4 viewProjectionMatrix = projectionMatrix * viewMatrix;
	UpdateCameraData(camPos, projectionMatrix, viewMatrix);

	//SceneGraph Data
	SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pObjects = pSceneGraph->GetObjects();

	//TODO: create big coalesced memory array of vertex buffer(s)?
	for (const Mesh* pMesh : pObjects)
	{
		//Perform VertexShading
		VertexShader(pMesh, camPos, projectionMatrix, viewMatrix);
		//TODO: find out what order is best, for cudaDevCpy and Malloc

		cudaDeviceSynchronize();

		//Peform Rasterization
		const size_t numVertices = pMesh->GetVertices().size();
		const size_t numIndices = pMesh->GetIndexes().size();
		Rasterizer(numVertices, numIndices);

		cudaDeviceSynchronize();

		//TODO: texturing
		const Mesh::Textures& textures = pMesh->GetTextures();
		GPUTextures gpuTextures{};

		//Perform PixelShading
		PixelShader(gpuTextures, sampleState, isDepthColour);

		//cudaDeviceSynchronize(); does not need to wait for vertex shader
	}

	cudaDeviceSynchronize();

	//TODO: copy pixelshaderbuffer and present to window

	SwapBuffers();
	Present();
}

//-----CPU HELPER FUNCTIONS-----

CPU_CALLABLE void CUDARenderer::InitCUDARasterizer()
{
	const unsigned int width = m_WindowHelper.Width;
	const unsigned int height = m_WindowHelper.Height;

	//Allocate buffers
	cudaFree(m_dev_OVertexBuffer);
	cudaMalloc(&m_dev_OVertexBuffer, width * height * sizeof(OVertex));
	cudaMemset(m_dev_OVertexBuffer, 0, width * height * sizeof(OVertex));

	cudaFree(m_dev_PixelShaderBuffer);
	cudaMalloc(&m_dev_PixelShaderBuffer, width * height * sizeof(OVertex));
	cudaMemset(m_dev_PixelShaderBuffer, 0, width * height * sizeof(OVertex));

	cudaFree(m_dev_FrameBuffer);
	cudaMalloc(&m_dev_FrameBuffer, width * height * sizeof(RGBColor));
	cudaMemset(m_dev_FrameBuffer, 0, width * height * sizeof(RGBColor));

	cudaFree(m_dev_DepthBuffer);
	cudaMalloc(&m_dev_DepthBuffer, width * height * sizeof(float));
	cudaMemset(m_dev_DepthBuffer, 0, width * height * sizeof(float));

	cudaFree(m_dev_RenderData);
	cudaMalloc(&m_dev_RenderData, sizeof(RenderData));
	cudaMemset(m_dev_RenderData, 0, sizeof(RenderData));

	//TODO: pin memory
	//cudaHostAlloc();

	//checkCUDAError("rasterizeInit");
}

CPU_CALLABLE void CUDARenderer::AllocateMeshBuffers(const size_t numVertices, const size_t numIndices)
{
	cudaFree(m_dev_IVertexBuffer);
	cudaMalloc(&m_dev_IVertexBuffer, numVertices * sizeof(IVertex));

	cudaFree(m_dev_IndexBuffer);
	cudaMalloc(&m_dev_IndexBuffer, numIndices * sizeof(unsigned int));

	cudaFree(m_dev_OVertexBuffer);
	cudaMalloc(&m_dev_OVertexBuffer, numVertices * sizeof(OVertex));
}

CPU_CALLABLE void CUDARenderer::CopyMeshBuffers(const std::vector<IVertex>& vertexBuffer, const std::vector<unsigned int>& indexBuffer)
{
	cudaMemcpy(m_dev_IVertexBuffer, vertexBuffer.data(), vertexBuffer.size() * sizeof(IVertex), cudaMemcpyHostToDevice);
	cudaMemcpy(m_dev_IndexBuffer, indexBuffer.data(), indexBuffer.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
}

CPU_CALLABLE void CUDARenderer::FreeMeshBuffers()
{
	cudaFree(m_dev_IndexBuffer);
	cudaFree(m_dev_IVertexBuffer);
	cudaFree(m_dev_OVertexBuffer);
}

CPU_CALLABLE void CUDARenderer::FreeCUDARasterizer()
{
	//Free buffers
	cudaFree(m_dev_IVertexBuffer);
	m_dev_IVertexBuffer = nullptr;

	cudaFree(m_dev_IndexBuffer);
	m_dev_IndexBuffer = nullptr;

	cudaFree(m_dev_OVertexBuffer);
	m_dev_OVertexBuffer = nullptr;

	cudaFree(m_dev_PixelShaderBuffer);
	m_dev_PixelShaderBuffer = nullptr;

	cudaFree(m_dev_FrameBuffer);
	m_dev_FrameBuffer = nullptr;

	cudaFree(m_dev_DepthBuffer);
	m_dev_DepthBuffer = nullptr;

	cudaFree(m_dev_RenderData);
	m_dev_RenderData = nullptr;


	//checkCUDAError("rasterize Free");
}

CPU_CALLABLE void CUDARenderer::UpdateCameraData(const FPoint3& camPos, const FMatrix4& projectionMatrix, const FMatrix4& viewMatrix)
{
	const size_t renderDataSizeBytes = sizeof(RenderData) - sizeof(FMatrix4); //without worldmatrix data
	RenderData renderDataBuffer;
	memcpy(&renderDataBuffer.camPos, &camPos, sizeof(camPos));
	memcpy(&renderDataBuffer.projectionMatrix, &projectionMatrix, sizeof(projectionMatrix));
	memcpy(&renderDataBuffer.viewMatrix, &viewMatrix, sizeof(viewMatrix));
	cudaMemcpy(m_dev_RenderData, &renderDataBuffer, renderDataSizeBytes, cudaMemcpyHostToDevice);
}

CPU_CALLABLE void CUDARenderer::UpdateWorldMatrixData(const FMatrix4& worldMatrix)
{
	const size_t pointerOffsetInBytes = sizeof(RenderData) - sizeof(FMatrix4);
	cudaMemcpy(m_dev_RenderData + pointerOffsetInBytes, &worldMatrix, sizeof(worldMatrix), cudaMemcpyHostToDevice);
}

//-----Misc HELPER FUNCTIONS-----

BOTH_CALLABLE float GetMinElement(float val0, float val1, float val2)
{
	float min = val0;
	if (val1 < min)
		min = val1;
	if (val2 < min)
		min = val2;
	return min;
}

BOTH_CALLABLE float GetMaxElement(float val0, float val1, float val2)
{
	float max = val0;
	if (val1 > max)
		max = val1;
	if (val2 > max)
		max = val2;
	return max;
}

//-----GPU HELPER FUNCTIONS-----

GPU_CALLABLE OVertex GetNDCVertex(const IVertex& vertex, const FPoint3& camPos,
	const FMatrix4& viewMatrix, const FMatrix4& projectionMatrix, const FMatrix4& worldMatrix)
{
	const FMatrix4 worldViewProjectionMatrix = projectionMatrix * viewMatrix * worldMatrix;

	FPoint4 NDCspace = worldViewProjectionMatrix * FPoint4{ vertex.v.x, vertex.v.y, vertex.v.z, vertex.v.z };
	NDCspace.x /= NDCspace.w;
	NDCspace.y /= NDCspace.w;
	NDCspace.z /= NDCspace.w;

	const FPoint3 worldPosition{ worldMatrix * FPoint4{ vertex.v } };
	const FVector3 viewDirection{ GetNormalized(worldPosition - camPos) };
	const FVector3 worldNormal{ worldMatrix * FVector4{ vertex.n } };
	const FVector3 worldTangent{ worldMatrix * FVector4{ vertex.tan } };

	return OVertex{ NDCspace, vertex.uv, worldNormal, worldTangent, vertex.c, viewDirection };
}

GPU_CALLABLE bool EdgeFunction(const FVector2& v0, const FVector2& v1, const FPoint2& pixel)
{
	// counter-clockwise
	const FVector2 edge{ v0 - v1 };
	const FVector2 vertexToPixel{ pixel - v0 };
	const float cross = Cross(edge, vertexToPixel);
	//TODO: weight? (totalArea..)
	return cross < 0.f;
}

GPU_CALLABLE bool IsPixelInTriangle(FPoint4 rasterCoords[3], const FPoint2& pixel, float weights[3])
{
	const FPoint2& v0 = rasterCoords[0].xy;
	const FPoint2& v1 = rasterCoords[1].xy;
	const FPoint2& v2 = rasterCoords[2].xy;

	const FVector2 edgeA{ v0 - v1 };
	const FVector2 edgeB{ v1 - v2 };
	const FVector2 edgeC{ v2 - v0 };
	// counter-clockwise

	bool isInTriangle{ true };
	const float totalArea = Cross(edgeA, edgeC);

	// edgeA
	FVector2 vertexToPixel{ pixel - v0 };
	float cross = Cross(edgeA, vertexToPixel);
	isInTriangle &= cross < 0.f;
	// weight2 == positive cross of 'previous' edge, for v2 this is edgeA (COUNTER-CLOCKWISE)
	weights[2] = cross / totalArea;

	// edgeB
	vertexToPixel = { pixel - v1 };
	cross = Cross(edgeB, vertexToPixel);
	isInTriangle &= cross < 0.f;
	// weight1 (for v1 this is edgeB)
	weights[1] = cross / totalArea;

	// edgeC
	vertexToPixel = { pixel - v2 };
	cross = Cross(edgeC, vertexToPixel);
	isInTriangle &= cross < 0.f;
	// weight0 (for v0 this is edgeC)
	weights[0] = cross / totalArea;

	//weights == inverted negative cross of 'previous' edge
	//weights[0] = Cross(-vertexToPixel, edgeC) / totalArea;
	//weights[1] = Cross(-vertexToPixel, edgeB) / totalArea;
	//weights[2] = Cross(-vertexToPixel, edgeA) / totalArea;
	// gives positive results because counter-clockwise
	//const float total = weights[0] + weights[1] + weights[2]; // total result equals 1

	//TODO use?
	//isInTriangle &= EdgeFunction((FVector2)v0, (FVector2)v1, pixel); //edgeA
	//isInTriangle &= EdgeFunction((FVector2)v1, (FVector2)v2, pixel); //edgeB
	//isInTriangle &= EdgeFunction((FVector2)v2, (FVector2)v0, pixel); //edgeC

	return isInTriangle;
}

GPU_CALLABLE bool DepthTest(FPoint4 rasterCoords[3], float& depthBuffer, float weights[3], float& zInterpolated)
{
	zInterpolated = (weights[0] * rasterCoords[0].z) + (weights[1] * rasterCoords[1].z) + (weights[2] * rasterCoords[2].z);

	if (zInterpolated < 0.f || zInterpolated > 1.f) return false;
	if (zInterpolated > depthBuffer) return false;

	depthBuffer = zInterpolated;

	return true;
}

GPU_CALLABLE bool FrustumTestVertex(const OVertex& NDC)
{
	bool isPassed{ false };
	isPassed &= (NDC.v.x < -1.f || NDC.v.x > 1.f); // perspective divide X in NDC
	isPassed &= (NDC.v.y < -1.f || NDC.v.y > 1.f); // perspective divide Y in NDC
	isPassed &= (NDC.v.z < 0.f || NDC.v.z > 1.f); // perspective divide Z in NDC
	return isPassed;
}

GPU_CALLABLE bool FrustumTest(const OVertex* NDC[3])
{
	bool isPassed{ false };
	isPassed &= FrustumTestVertex(*NDC[0]);
	isPassed &= FrustumTestVertex(*NDC[1]);
	isPassed &= FrustumTestVertex(*NDC[2]);
	return isPassed;
}

GPU_CALLABLE void NDCToScreenSpace(FPoint4 rasterCoords[3], const unsigned int width, const unsigned int height)
{
	for (int i{}; i < 3; ++i)
	{
		rasterCoords[i].x = ((rasterCoords[i].x + 1) / 2) * width;
		rasterCoords[i].y = ((1 - rasterCoords[i].y) / 2) * height;
	}
}

GPU_CALLABLE BoundingBox GetBoundingBox(FPoint4 rasterCoords[3], const unsigned int width, const unsigned int height)
{
	BoundingBox bb;
	bb.xMin = (short)GetMinElement(rasterCoords[0].x, rasterCoords[1].x, rasterCoords[2].x) - 1; // xMin
	bb.yMin = (short)GetMinElement(rasterCoords[0].y, rasterCoords[1].y, rasterCoords[2].y) - 1; // yMin
	bb.xMax = (short)GetMaxElement(rasterCoords[0].x, rasterCoords[1].x, rasterCoords[2].x) + 1; // xMax
	bb.yMax = (short)GetMaxElement(rasterCoords[0].y, rasterCoords[1].y, rasterCoords[2].y) + 1; // yMax

	if (bb.xMin < 0) bb.xMin = 0; //clamp minX to Left of screen
	if (bb.yMin < 0) bb.yMin = 0; //clamp minY to Bottom of screen
	if (bb.xMax > width) bb.xMax = width; //clamp maxX to Right of screen
	if (bb.yMax > height) bb.yMax = height; //clamp maxY to Top of screen

	return bb;
}

GPU_CALLABLE GPU_INLINE RGBColor ShadePixel(const OVertex& oVertex, const GPUTextures& textures, SampleState sampleState, bool isDepthColour)
{
	RGBColor finalColour{};
	if (isDepthColour)
	{
		finalColour = RGBColor{ Remap(oVertex.v.z, 0.985f, 1.f), 0.f, 0.f }; // depth colour
		finalColour.ClampColor();
	}
	else
	{
		const RGBColor diffuseColour = GPUTextureSampler::Sample(textures.pDiff, oVertex.uv, sampleState);

		const RGBColor normalRGB = GPUTextureSampler::Sample(textures.pNorm, oVertex.uv, sampleState);
		FVector3 normal{ normalRGB.r, normalRGB.g, normalRGB.b };

		FVector3 binormal{ Cross(oVertex.tan, oVertex.n) };
		FMatrix3 tangentSpaceAxis{ oVertex.tan, binormal, oVertex.n };

		normal.x = 2.f * normal.x - 1.f;
		normal.y = 2.f * normal.y - 1.f;
		normal.z = 2.f * normal.z - 1.f;

		normal = tangentSpaceAxis * normal;

		//// light calculations
		//for (Light* pLight : sm.GetSceneGraph()->GetLights())
		//{
		//	const FVector3& lightDir{ pLight->GetDirection(FPoint3{}) };
		//	const float observedArea{ Dot(-normal, lightDir) };
		//
		//	if (observedArea < 0.f)
		//		continue;
		//
		//	const RGBColor biradiance{ pLight->GetBiradiance(FPoint3{}) };
		//	// swapped direction of lights
		//
		//	// phong
		//	const FVector3 reflectV{ Reflect(lightDir, normal) };
		//	//Normalize(reflectV);
		//	const float angle{ Dot(reflectV, oVertex.vd) };
		//	const RGBColor specularSample{ textures.pSpec->Sample(oVertex.uv, sampleState) };
		//	const RGBColor phongSpecularReflection{ specularSample * powf(angle, textures.pGloss->Sample(oVertex.uv, sampleState).r * 25.f) };
		//
		//	//const RGBColor lambertColour{ diffuseColour * (RGBColor{1.f,1.f,1.f} - specularSample) };
		//	//const RGBColor lambertColour{ (diffuseColour / float(E_PI)) * (RGBColor{1.f,1.f,1.f} - specularSample) };
		//	const RGBColor lambertColour{ (diffuseColour * specularSample) / float(E_PI) }; //severely incorrect result, using diffusecolour for now
		//	// Lambert diffuse == incoming colour multiplied by diffuse coefficient (1 in this case) divided by Pi
		//	finalColour += biradiance * (diffuseColour + phongSpecularReflection) * observedArea;
		//}
		finalColour.ClampColor();
	}
	return finalColour;
}

GPU_CALLABLE void RenderPixelsInTriangle(OVertex* pPixelShaderBuffer, OVertex* screenspaceTriangle[3], FPoint4 rasterCoords[3], float* pDepthBuffer,
	const BoundingBox& bb, const unsigned int width, const unsigned int height)
{
	const OVertex& v0 = *screenspaceTriangle[0];
	const OVertex& v1 = *screenspaceTriangle[1];
	const OVertex& v2 = *screenspaceTriangle[2];

	//Loop over all pixels in bounding box
	for (uint32_t r = bb.yMin; r < bb.yMax; ++r)
	{
		for (uint32_t c = bb.xMin; c < bb.xMax; ++c)
		{
			const unsigned int pixelIdx = c + r * width;
			OVertex oVertex{};
			const FPoint2 pixel{ float(c), float(r) };
			float weights[3];
			if (IsPixelInTriangle(rasterCoords, pixel, weights))
			{
				float zInterpolated{};
				if (DepthTest(rasterCoords, pDepthBuffer[pixelIdx], weights, zInterpolated))
				{
					const float wInterpolated = (weights[0] * v0.v.w) + (weights[1] * v1.v.w) + (weights[2] * v2.v.w);

					FVector2 interpolatedUV{
						weights[0] * (v0.uv.x / rasterCoords[0].w) + weights[1] * (v1.uv.x / rasterCoords[1].w) + weights[2] * (v2.uv.x / rasterCoords[2].w),
						weights[0] * (v0.uv.y / rasterCoords[0].w) + weights[1] * (v1.uv.y / rasterCoords[1].w) + weights[2] * (v2.uv.y / rasterCoords[2].w) };
					interpolatedUV *= wInterpolated;

					FVector3 interpolatedNormal{
						 weights[0] * (v0.n.x / rasterCoords[0].w) + weights[1] * (v1.n.x / rasterCoords[1].w) + weights[2] * (v2.n.x / rasterCoords[2].w),
						 weights[0] * (v0.n.y / rasterCoords[0].w) + weights[1] * (v1.n.y / rasterCoords[1].w) + weights[2] * (v2.n.y / rasterCoords[2].w),
						 weights[0] * (v0.n.z / rasterCoords[0].w) + weights[1] * (v1.n.z / rasterCoords[1].w) + weights[2] * (v2.n.z / rasterCoords[2].w) };
					interpolatedNormal *= wInterpolated;

					const FVector3 interpolatedTangent{
						weights[0] * (v0.tan.x / rasterCoords[0].w) + weights[1] * (v1.tan.x / rasterCoords[1].w) + weights[2] * (v2.tan.x / rasterCoords[2].w),
						weights[0] * (v0.tan.y / rasterCoords[0].w) + weights[1] * (v1.tan.y / rasterCoords[1].w) + weights[2] * (v2.tan.y / rasterCoords[2].w),
						weights[0] * (v0.tan.z / rasterCoords[0].w) + weights[1] * (v1.tan.z / rasterCoords[1].w) + weights[2] * (v2.tan.z / rasterCoords[2].w) };

					FVector3 interpolatedViewDirection{
					weights[0] * (v0.vd.y / rasterCoords[0].w) + weights[1] * (v1.vd.y / rasterCoords[1].w) + weights[2] * (v2.vd.y / rasterCoords[2].w),
					weights[0] * (v0.vd.x / rasterCoords[0].w) + weights[1] * (v1.vd.x / rasterCoords[1].w) + weights[2] * (v2.vd.x / rasterCoords[2].w),
					weights[0] * (v0.vd.z / rasterCoords[0].w) + weights[1] * (v1.vd.z / rasterCoords[1].w) + weights[2] * (v2.vd.z / rasterCoords[2].w) };
					Normalize(interpolatedViewDirection);

					const RGBColor interpolatedColour{
						weights[0] * (v0.c.r / rasterCoords[0].w) + weights[1] * (v1.c.r / rasterCoords[1].w) + weights[2] * (v2.c.r / rasterCoords[2].w),
						weights[0] * (v0.c.g / rasterCoords[0].w) + weights[1] * (v1.c.g / rasterCoords[1].w) + weights[2] * (v2.c.g / rasterCoords[2].w),
						weights[0] * (v0.c.b / rasterCoords[0].w) + weights[1] * (v1.c.b / rasterCoords[1].w) + weights[2] * (v2.c.b / rasterCoords[2].w) };

					oVertex.v = FPoint4{ pixel, zInterpolated, wInterpolated };
					oVertex.c = std::move(interpolatedColour);
					oVertex.uv = std::move(interpolatedUV);
					oVertex.n = std::move(interpolatedNormal);
					oVertex.tan = std::move(interpolatedTangent);
					oVertex.vd = std::move(interpolatedViewDirection);
					
				}
			}
			pPixelShaderBuffer[pixelIdx] = std::move(oVertex);
		}
	}
}

GPU_CALLABLE void RasterizeTriangle(OVertex* pPixelShaderBuffer, OVertex* triangle[3], FPoint4 rasterCoords[3], 
	float* pDepthBuffer, const unsigned int width, const unsigned int height)
{
	NDCToScreenSpace(rasterCoords, width, height);
	const BoundingBox bb = GetBoundingBox(rasterCoords, width, height);
	RenderPixelsInTriangle(pPixelShaderBuffer, triangle, rasterCoords, pDepthBuffer, bb, width, height);
}

GPU_CALLABLE void InitDepth(float* pDepthBuffer, const unsigned int width, const unsigned int height)
{
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < width && y < height)
	{
		const unsigned int index = x + (y * width);
		pDepthBuffer[index] = FLT_MAX;
	}
}

//-----KERNELS-----
//Kernel launch params:	numBlocks, numThreadsPerBlock, numSharedMemoryBytes, stream

GPU_KERNEL void ResetDepthBuffer(float* pDepthBuffer, const unsigned int width, const unsigned int height)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int pixelIdx = x + y * width;
	if (x < width && y < height)
	{
		pDepthBuffer[pixelIdx] = INT_MAX;
	}
}

//value of 64 for R,G,B colour of dark grey
GPU_KERNEL void ClearFrameBuffer(RGBColor* pFrameBuffer, const unsigned int width, const unsigned int height, const RGBColor& colour)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int pixelIdx = x + y * width;
	if (x < width && y < height)
	{
		pFrameBuffer[pixelIdx] = colour;
	}
}

GPU_KERNEL void VertexShaderKernel(IVertex IVertices[], OVertex OVertices[], const size_t size,
	const FPoint3 camPos, const FMatrix4 projectionMatrix, const FMatrix4 viewMatrix, const FMatrix4 worldMatrix)
{
	GPU_SHARED_MEMORY float renderData[sizeof(CUDARenderer::RenderData)];

	const FPoint3 camPosShared;
	const FMatrix4 projectionMatrixShared{};
	const FMatrix4 viewMatrixShared{};
	const FMatrix4 worldMatrixShared{};

	// vertex id
	const unsigned int vIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vIdx < size)
	{
		OVertices[vIdx] = GetNDCVertex(IVertices[vIdx], camPos, viewMatrix, projectionMatrix, worldMatrix);
	}

	__syncthreads();
}

GPU_KERNEL void RasterizerKernel(OVertex* pPixelShaderBuffer, OVertex* pTransformedVertices, const size_t numVertices, unsigned int* pIndexBuffer, const size_t numIndices,
	float* pDepthBuffer, const unsigned int width, const unsigned int height)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	//const unsigned int idx = (x + pixelXoffset) + (y + pixelYoffset) * imageWidth;
	const unsigned int idx = x + y * width;

	if (idx < numVertices)
	{
		OVertex& v0 = pTransformedVertices[pIndexBuffer[idx - 2]];
		OVertex& v1 = pTransformedVertices[pIndexBuffer[idx - 1]];
		OVertex& v2 = pTransformedVertices[pIndexBuffer[idx]];

		OVertex* triangle[3]{ &v0, &v1, &v2 };
		FPoint4 rasterCoords[3]{ v0.v, v1.v, v2.v };

		NDCToScreenSpace(rasterCoords, width, height);
		const BoundingBox bb = GetBoundingBox(rasterCoords, width, height);
		//Rasterize Screenspace triangle
		RenderPixelsInTriangle(pPixelShaderBuffer, triangle, rasterCoords, pDepthBuffer, bb, width, height);
	}
	__syncthreads();

	/*
	Send 1 triangle per thread (coarse rasterizer)

	thread calculates 3 vertices, barycentric coordinates

	thread atomically does depthtest in rasterizer stage before pixelshader

	TODO: use shared memory, then coalescened copy
	e.g. single bin buffer in single shared memory

	TODO: use binning, each bin their AABBs (and checks) (bin rasterizer)
	*/
}

GPU_KERNEL void PixelShaderKernel(RGBColor* pFrameBuffer, OVertex* pPixelShaderBuffer, GPUTextures textures, SampleState sampleState, bool isDepthColour, const unsigned int width)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	const unsigned int pixelIdx = x + y * width;

	OVertex oVertex = pPixelShaderBuffer[pixelIdx];
	RGBColor colour = ShadePixel(oVertex, textures, sampleState, isDepthColour);
	pFrameBuffer[pixelIdx] = colour;

	//__syncthreads();
}

//-----KERNEL LAUNCHERS-----

CPU_CALLABLE void CUDARenderer::Clear(const RGBColor& colour)
{
	ResetDepthBuffer<<<m_NumBlocks, m_NumThreadsPerBlock>>>
		(m_dev_DepthBuffer, m_WindowHelper.Width, m_WindowHelper.Height);
	ClearFrameBuffer<<<m_NumBlocks, m_NumThreadsPerBlock>>>
		(m_dev_FrameBuffer, m_WindowHelper.Width, m_WindowHelper.Height, colour);
}

CPU_CALLABLE void CUDARenderer::VertexShader(const Mesh* pMesh, const FPoint3& camPos, const FMatrix4& projectionMatrix, const FMatrix4& viewMatrix)
{
	const FMatrix4& worldMatrix = pMesh->GetWorldMatrix();
	UpdateWorldMatrixData(worldMatrix);

	const std::vector<IVertex>& vertexBuffer = pMesh->GetVertices();
	const std::vector<unsigned int>& indexBuffer = pMesh->GetIndexes();
	const size_t numVertices = vertexBuffer.size();
	const size_t numIndices = indexBuffer.size();

	//TODO: memory pool???
	FreeMeshBuffers(); //free device buffers
	AllocateMeshBuffers(numVertices, numIndices); //allocate device buffers
	CopyMeshBuffers(vertexBuffer, indexBuffer); //copy host memory into device memory

	//TODO: async & streams
	//cudaMallocAsync();

	const unsigned int numBlocksForVertices = (numVertices + m_NumThreadsPerBlock.x - 1) / m_NumThreadsPerBlock.x;
	const size_t numSharedMemoryBytes = sizeof(RenderData);
	VertexShaderKernel << <numBlocksForVertices, m_NumThreadsPerBlock, numSharedMemoryBytes >> >
		(m_dev_IVertexBuffer, m_dev_OVertexBuffer, numVertices,
			camPos, projectionMatrix, viewMatrix, worldMatrix);
}

CPU_CALLABLE void CUDARenderer::Rasterizer(const size_t numVertices, const size_t numIndices)
{
	//TODO: different launch params
	RasterizerKernel<<<m_NumBlocks, m_NumThreadsPerBlock>>>
		(m_dev_PixelShaderBuffer, m_dev_OVertexBuffer, numVertices, m_dev_IndexBuffer, numIndices,
			m_WindowHelper.pDepthBuffer, m_WindowHelper.Width, m_WindowHelper.Height);
}

CPU_CALLABLE void CUDARenderer::PixelShader(const GPUTextures& textures, SampleState sampleState, bool isDepthColour)
{
	PixelShaderKernel<<<m_NumBlocks, m_NumThreadsPerBlock>>>
		(m_dev_FrameBuffer, m_dev_PixelShaderBuffer, textures, sampleState, isDepthColour, m_WindowHelper.Width);
}