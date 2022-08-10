#include "PCH.h"
#include "CUDARenderer.cuh"
#include <vector>

#include "DEFINES.h"

//Project CUDA includes
#include "GPUTextureSampler.cuh"
#include "CUDAMatrixMath.cuh"

//Project includes
#include "WindowHelper.h"
#include "SceneManager.h"
#include "SceneGraph.h"
#include "Mesh.h"
#include "Camera.h"
#include "Vertex.h"
#include "PrimitiveTopology.h"
#include "BoundingBox.h"
#include "GPUTextures.h"
#include "RGBRaw.h"
#include "Light.h"
#include "GPUTextures.h"
#include "CullingMode.h"
#include "CUDABenchMarker.h"
#include "MemoryConversionStrings.h"

#pragma region STRUCT DECLARATIONS

struct RenderData // size == 35
{
	union
	{
		float pData[];
		struct
		{
			FPoint3 CamPos;
			FMatrix4 ViewProjectionMatrix;
			FMatrix4 WorldMatrix;
		};
	};
};

struct RasterTriangle // size == 12
{
	FPoint4 v0;
	FPoint4 v1;
	FPoint4 v2;
};

struct TriangleIdx // size == 3
{
	union
	{
		unsigned int idx[];
		struct
		{
			unsigned int idx0;
			unsigned int idx1;
			unsigned int idx2;
		};
	};
	//bool isCulled; //padded anyway
};

struct TriangleIdxBb // size == 5
{
	union
	{
		unsigned int idx[];
		struct
		{
			unsigned int idx0;
			unsigned int idx1;
			unsigned int idx2;
		};
	};
	BoundingBox bb;
};

struct PixelShade // size == 32
{
	unsigned int colour;
	float zInterpolated;
	float wInterpolated;
	FVector2 uv;
	FVector3 n;
	FVector3 tan;
	FVector3 vd;
	GPUTexturesCompact textures; // size == 18
};

struct CUDAMeshBuffers // size == 12
{
	FPoint4* pPositions;
	union
	{
		OVertexData* pVertexDatas;
		struct
		{
			FVector2* pUVs;
			FVector3* pNormals;
			FVector3* pTangents;
			FVector3* pViewDirections;
			RGBColor* pColours;
		};
	};
};

#pragma endregion

#pragma region GLOBAL VARIABLES

constexpr int NumTextures = 4;

//CONST DEVICE MEMORY - Does NOT have to be allocated or freed
GPU_CONST_MEMORY static float dev_RenderData_const[sizeof(RenderData) / sizeof(float)]{};
GPU_CONST_MEMORY float dev_CameraPos_const[sizeof(FPoint3) / sizeof(float)]{};
GPU_CONST_MEMORY float dev_WVPMatrix_const[sizeof(FMatrix4) / sizeof(float)]{};
GPU_CONST_MEMORY float dev_WorldMatrix_const[sizeof(FMatrix4) / sizeof(float)]{};
GPU_CONST_MEMORY float dev_RotationMatrix_const[sizeof(FMatrix3) / sizeof(float)]{};
//NOTE: cannot contain anything else besides primitive variables (int, float, etc.)

//DEVICE MEMORY - Does have to be allocated and freed
static unsigned int* dev_FrameBuffer{};
static int* dev_DepthBuffer{}; //defined as INTEGER type for atomicCAS to work properly
static int* dev_MutexBuffer{};
static PixelShade* dev_PixelShadeBuffer{}; //(== fragmentbuffer)
static std::vector<IVertex*> dev_IVertexBuffer{};
static std::vector<unsigned int*> dev_IndexBuffer{};
//static std::vector<CUDAMeshBuffers*> dev_OMeshBuffers{};
static std::vector<OVertex*> dev_OVertexBuffer{};
static std::vector<TriangleIdx*> dev_Triangles{};
static std::vector<unsigned int*> dev_TextureData{};
/*
//DEPRECATED
//Texture references have to be statically declared in global memory and bound to CUDA texture memory
//They cannot be referenced in functions, nor used in arrays
typedef texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> CUDA32bTexture2D;
static CUDA32bTexture2D dev_DiffuseTextureReference{};
static CUDA32bTexture2D dev_NormalTextureReference{};
static CUDA32bTexture2D dev_SpecularTextureReference{};
static CUDA32bTexture2D dev_GlossinessTextureReference{};
*/

#pragma endregion

//--------------------------

CPU_CALLABLE
CUDARenderer::CUDARenderer(const WindowHelper& windowHelper)
	: m_WindowHelper{ windowHelper }
	, m_TotalNumTriangles{}
	, m_TotalVisibleNumTriangles{}
	, m_TimerMs{}
	, m_h_pFrameBuffer{}
	, m_MeshIdentifiers{}
	, m_TextureObjects{}
{
	InitCUDADeviceBuffers();
	CheckErrorCuda(cudaEventCreate(&m_StartEvent));
	CheckErrorCuda(cudaEventCreate(&m_StopEvent));
}

CPU_CALLABLE
CUDARenderer::~CUDARenderer()
{
	CheckErrorCuda(DeviceSynchroniseCuda());
	CheckErrorCuda(cudaEventDestroy(m_StartEvent));
	CheckErrorCuda(cudaEventDestroy(m_StopEvent));
	FreeCUDADeviceBuffers();
}

#pragma region MISC HELPER FUNCTIONS

BOTH_CALLABLE static
float GetMinElement(float val0, float val1, float val2)
{
	float min = val0;
	if (val1 < min)
		min = val1;
	if (val2 < min)
		min = val2;
	return min;
}

BOTH_CALLABLE static
float GetMaxElement(float val0, float val1, float val2)
{
	float max = val0;
	if (val1 > max)
		max = val1;
	if (val2 > max)
		max = val2;
	return max;
}

#pragma endregion

#pragma region CPU HELPER FUNCTIONS

#pragma region PUBLIC FUNCTIONS

CPU_CALLABLE
void CUDARenderer::LoadScene(const SceneGraph* pSceneGraph)
{
	if (!pSceneGraph)
	{
		std::cout << "!CUDARenderer::LoadScene > Invalid scenegraph!\n";
		return;
	}
	m_TotalNumTriangles = 0;
	FreeMeshBuffers();
	const std::vector<Mesh*>& pMeshes = pSceneGraph->GetMeshes();
	for (const Mesh* pMesh : pMeshes)
	{
		MeshIdentifier mi{};
		mi.Idx = m_MeshIdentifiers.size();
		mi.pMesh = pMesh;
		size_t numTriangles{};

		float* vertexBuffer = pMesh->GetVertices();
		unsigned int* indexBuffer = pMesh->GetIndexes();
		const unsigned int numVertices = pMesh->GetVertexAmount();
		const unsigned int numIndices = pMesh->GetIndexAmount();
		const PrimitiveTopology topology = pMesh->GetTopology();
		const short vertexType = pMesh->GetVertexType();
		const short stride = pMesh->GetVertexStride();
		const FMatrix4& worldMat = pMesh->GetWorldMatrix();

		switch (topology)
		{
		case PrimitiveTopology::TriangleList:
			numTriangles += numIndices / 3;
			break;
		case PrimitiveTopology::TriangleStrip:
			numTriangles += numIndices - 2;
			break;
		}
		mi.TotalNumTriangles = numTriangles;

		AllocateMeshBuffers(numVertices, numIndices, numTriangles, mi.Idx);
		CopyMeshBuffers(vertexBuffer, numVertices, stride, indexBuffer, numIndices, mi.Idx);
		LoadMeshTextures(pMesh->GetTexPaths(), mi.Idx);
		mi.Textures = m_TextureObjects[mi.Idx];

		m_TotalNumTriangles += numTriangles;
		m_MeshIdentifiers.push_back(mi);
	}
}

CPU_CALLABLE
void CUDARenderer::Render(const SceneManager& sm, const Camera* pCamera)
{
	//Render Data
	const bool isDepthColour = sm.IsDepthColour();
	const SampleState sampleState = sm.GetSampleState();
	const CullingMode cm = sm.GetCullingMode();

	//Camera Data
	const FPoint3& camPos = pCamera->GetPos();
	const FVector3& camFwd = pCamera->GetForward();
	const FMatrix4 lookatMatrix = pCamera->GetLookAtMatrix();
	const FMatrix4 viewMatrix = pCamera->GetViewMatrix(lookatMatrix);
	const FMatrix4 projectionMatrix = pCamera->GetProjectionMatrix();
	const FMatrix4 viewProjectionMatrix = projectionMatrix * viewMatrix;

	//TODO: use renderdata as constant memory

	//TODO: random illegal memory access BUG
	//Update global memory for camera's matrices
	//UpdateCameraDataAsync(camPos, viewProjectionMatrix);

	CheckErrorCuda(cudaMemcpyToSymbol(dev_CameraPos_const, camPos.data, sizeof(camPos), 0, cudaMemcpyHostToDevice));

	//SceneGraph Data
	const SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pObjects = pSceneGraph->GetMeshes();

#ifdef BENCHMARK
	float vertexShadingMs{};
	float TriangleAssemblingMs{};
	float RasterizationMs{};
#endif

	m_TotalVisibleNumTriangles = 0;
	for (MeshIdentifier& mi : m_MeshIdentifiers)
	{
		//Mesh Data
		const Mesh* pMesh = pObjects[mi.Idx];
		const FMatrix4& worldMat = pMesh->GetWorldMatrix();
		const FMatrix4 worldViewProjectionMatrix = viewProjectionMatrix * worldMat;
		const FMatrix3 rotationMatrix = (FMatrix3)worldMat;

		//Update const data
		CheckErrorCuda(cudaMemcpyToSymbol(dev_WorldMatrix_const, worldMat.data, sizeof(worldMat), 0, cudaMemcpyHostToDevice));
		CheckErrorCuda(cudaMemcpyToSymbol(dev_WVPMatrix_const, worldViewProjectionMatrix.data, sizeof(worldViewProjectionMatrix), 0, cudaMemcpyHostToDevice));
		CheckErrorCuda(cudaMemcpyToSymbol(dev_RotationMatrix_const, rotationMatrix.data, sizeof(rotationMatrix), 0, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		//TODO: can async copy (parts of) mesh buffers H2D

#ifdef BENCHMARK
		StartTimer();
#endif
		//TODO: async & streams
		//TODO: find out what order is best, for cudaDevCpy and Malloc
		//---STAGE 1---:  Perform Output Vertex Assembling
		VertexShader(mi);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 1---
#ifdef BENCHMARK
		vertexShadingMs += StopTimer();
		StartTimer();
#endif
		//---STAGE 2---:  Perform Triangle Assembling
		TriangleAssembler(mi);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 2---
#ifdef BENCHMARK
		TriangleAssemblingMs += StopTimer();
		StartTimer();
#endif
		//---STAGE 3---: Peform Triangle Rasterization & Pixel Shading
		Rasterizer(mi, camFwd, cm);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 3---
#ifdef BENCHMARK
		RasterizationMs += StopTimer();
#endif
	}

#ifdef BENCHMARK
	StartTimer();
#endif
	//---STAGE 4---: Peform  Pixel Shading
	PixelShader(sampleState, isDepthColour);
	CheckErrorCuda(cudaDeviceSynchronize());
	//---END STAGE 4---
#ifdef BENCHMARK
	float PixelShadingMs = StopTimer();
	std::cout << "VS: " << vertexShadingMs << "ms | TA: " << TriangleAssemblingMs << "ms | Raster: " << RasterizationMs << "ms | PS: " << PixelShadingMs << "ms\r";
#endif
}

CPU_CALLABLE
void CUDARenderer::RenderAuto(const SceneManager& sm, const Camera* pCamera)
{
#ifdef _DEBUG
	if (EnterValidRenderingState())
		exit(1);
#else
	EnterValidRenderingState();
#endif

	Render(sm, pCamera);

	//TODO: parallel copies (streams & async)
	//Swap out buffers and update window
	Present();
}

CPU_CALLABLE
void CUDARenderer::StartTimer()
{
	CheckErrorCuda(cudaEventRecord(m_StartEvent));
}

CPU_CALLABLE
float CUDARenderer::StopTimer()
{
	CheckErrorCuda(cudaEventRecord(m_StopEvent));
	CheckErrorCuda(cudaEventSynchronize(m_StopEvent));
	CheckErrorCuda(cudaEventElapsedTime(&m_TimerMs, m_StartEvent, m_StopEvent));
	return m_TimerMs;
}

#pragma endregion

#pragma region PRIVATE FUNCTIONS

CPU_CALLABLE
void CUDARenderer::DisplayGPUSpecs(int deviceId)
{
	std::string yn{};

	std::cout << "\n---General---\n";
	cudaDeviceProp prop;
	CheckErrorCuda(cudaGetDeviceProperties(&prop, deviceId));
	std::cout << "Device detected: " << prop.name << '\n';
	std::cout << "Compute Capability: " << prop.major << '.' << prop.minor << '\n';
	std::cout << "Compute Mode: ";
	switch (prop.computeMode)
	{
	case cudaComputeModeDefault:
		std::cout << "Default\n";
		break;
	case cudaComputeModeExclusive:
		std::cout << "Exclusive\n";
		break;
	case cudaComputeModeProhibited:
		std::cout << "Prohibited\n";
		break;
	case cudaComputeModeExclusiveProcess:
		std::cout << "ExclusiveProcess\n";
		break;
	default:
		std::cout << "Undefined\n";
		break;
	}
	if (prop.isMultiGpuBoard)
	{
		std::cout << "Multi GPU setup: Yes\n";
		std::cout << "Multi GPU boardgroup ID: " << prop.multiGpuBoardGroupID << '\n';
	}
	else
	{
		std::cout << "Multi GPU setup: No\n";
	}
	std::cout << "Async Engine (DMA) count: " << prop.asyncEngineCount << '\n';
	yn = prop.deviceOverlap ? "Yes\n" : "No\n";
	std::cout << "Can concurrently copy memory between host and device while executing kernel: " << yn;
	switch (prop.asyncEngineCount)
	{
	case 0:
		std::cout << "Device cannot concurrently copy memory between host and device while executing a kernel\n";
		break;
	case 1:
		std::cout << "Device can concurrently copy memory between host and device while executing a kernel\n";
		break;
	case 2:
		std::cout << "Device can concurrently copy memory between host and device in both directions and execute a kernel at the same time\n";
		break;
	default:
		break;
	}
	yn = prop.concurrentKernels ? "Yes\n" : "No\n";
	std::cout << "Device supports executing multiple kernels within the same context simultaneously: " << yn;
	yn = prop.integrated ? "Yes\n" : "No\n";
	std::cout << "Integrated Graphics: " << yn;

	std::cout << "\n---Memory---\n";
	std::cout << "Total amount of Global Memory: " << ToMbs(prop.totalGlobalMem) << '\n';
	std::cout << "Total amount of Const Memory: " << ToKbs(prop.totalConstMem) << '\n';
	//size_t free{}, total{};
	//cudaMemGetInfo(&free, &total);
	//std::cout << "Total amount of VRAM: " << total << '\n';
	//std::cout << "Free amount of VRAM: " << free << '\n';
	std::cout << "Shared Memory per Multiprocessor: " << ToKbs(prop.sharedMemPerMultiprocessor) << '\n';
	std::cout << "Shared Memory per Block: " << ToKbs(prop.sharedMemPerBlock) << '\n';

	std::cout << "Shared Memory Reserved by CUDA driver per Block: " << prop.reservedSharedMemPerBlock << " bytes\n";

	yn = prop.unifiedAddressing ? "Yes\n" : "No\n";
	std::cout << "Unified Addressing supported: " << yn;
	yn = prop.managedMemory ? "Yes\n" : "No\n";
	std::cout << "Managed Memory supported: " << yn;
	yn = prop.pageableMemoryAccess ? "Yes\n" : "No\n";
	std::cout << "Device can coherently access Pageable Memory (non-pinned memory): " << yn;
	yn = prop.pageableMemoryAccessUsesHostPageTables ? "Yes\n" : "No\n";
	std::cout << "Device can access pageable memory via host's page tables: " << yn;
	yn = prop.canMapHostMemory ? "Yes\n" : "No\n";
	std::cout << "Can Map host memory: " << yn;

	std::cout << "\n---Memory - Caching---\n";
	yn = prop.globalL1CacheSupported ? "Yes\n" : "No\n";
	std::cout << "Global L1 Cache Supported: " << yn;
	yn = prop.localL1CacheSupported ? "Yes\n" : "No\n";
	std::cout << "Local L1 Cache Supported: " << yn;
	std::cout << "L2 Cache Size: " << ToKbs(prop.l2CacheSize) << '\n';;
	std::cout << "Persisting L2 Cache Max Size: " << prop.persistingL2CacheMaxSize << " bytes\n";

	std::cout << "\n---Other---\n";
	std::cout << "ClockRate: " << prop.clockRate / 1000 << "Khz\n";
	std::cout << "Memory ClockRate: " << prop.memoryClockRate / 1000 << "Khz\n";
	std::cout << "Memory Pitch: " << prop.memPitch << " bytes\n";
	std::cout << "Maximum number of 32-bit registers per Multiprocessor: " << prop.regsPerMultiprocessor << "\n";
	std::cout << "Maximum number of 32-bit registers per Block: " << prop.regsPerBlock << "\n";

	std::cout << "\n---Thread specifications---\n";
	std::cout << "Max threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << '\n';
	std::cout << "Max threads per Block: " << prop.maxThreadsPerBlock << '\n';
	std::cout << "Max threads Dimensions: X: " << prop.maxThreadsDim[0] << ", Y: " << prop.maxThreadsDim[1] << ", Z: " << prop.maxThreadsDim[2] <<'\n';
	std::cout << "Warp Size (in threads): " << prop.warpSize << '\n';
	std::cout << '\n';
}

CPU_CALLABLE
void CUDARenderer::InitCUDADeviceBuffers()
{
	DisplayGPUSpecs(0);

	size_t size{};
	const unsigned int width = m_WindowHelper.Width;
	const unsigned int height = m_WindowHelper.Height;

	//CUDAHOSTALLOC FLAGS
	/*
	cudaHostAllocDefault: This flag's value is defined to be 0 and causes cudaHostAlloc() to emulate cudaMallocHost().
	cudaHostAllocPortable: The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
	cudaHostAllocMapped: Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
	cudaHostAllocWriteCombined: Allocates the memory as write-combined (WC).
	WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read efficiently by most CPUs.
	WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
	*/

	//--->PINNED MEMORY<--- (HOST ONLY)
	//+ makes memory transactions between host and device significantly faster
	//- however this will allocate on host's RAM memory (in this case it would be 640 * 480 * 4 bytes == 1.2288Mb)
	size = sizeof(unsigned int);
	CheckErrorCuda(cudaMallocHost((void**)&m_h_pFrameBuffer, width * height * size));
	//CheckErrorCuda(cudaHostAlloc((void**)&m_h_pFrameBuffer, width * height * size, cudaHostAllocPortable));
	
	//host pinned memory without SDL window pixelbuffer
	//SDL allows random access to pixelbuffer, but cuda does not allowed host memory to be there

	//CUDAHOSTREGISTER FLAGS
	/*
	cudaHostRegisterDefault: On a system with unified virtual addressing, the memory will be both mapped and portable. 
	On a system with no unified virtual addressing, the memory will be neither mapped nor portable.
	cudaHostRegisterPortable: The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
	cudaHostRegisterMapped: Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
	cudaHostRegisterIoMemory: The passed memory pointer is treated as pointing to some memory-mapped I/O space, 
	e.g. belonging to a third-party PCIe device, and it will marked as non cache-coherent and contiguous.
	cudaHostRegisterReadOnly: The passed memory pointer is treated as pointing to memory that is considered read-only by the device.
	On platforms without cudaDevAttrPageableMemoryAccessUsesHostPageTables, this flag is required in order to register memory mapped to the CPU as read-only.
	Support for the use of this flag can be queried from the device attribute cudaDeviceAttrReadOnlyHostRegisterSupported.
	Using this flag with a current context associated with a device that does not have this attribute set will cause cudaHostRegister to error with cudaErrorNotSupported.
	*/

	//Invalid Argument
	//SDL_LockSurface(m_WindowHelper.pBackBuffer);
	//size = sizeof(unsigned int);
	//CheckErrorCuda(cudaHostRegister(m_WindowHelper.pBackBufferPixels, width * height * size, cudaHostRegisterDefault));
	//SDL_UnlockSurface(m_WindowHelper.pBackBuffer);;

	size = sizeof(PixelShade);
	CheckErrorCuda(cudaFree(dev_PixelShadeBuffer));
	CheckErrorCuda(cudaMalloc((void**)&dev_PixelShadeBuffer, width * height * size));
	CheckErrorCuda(cudaMemset(dev_PixelShadeBuffer, 0, width * height * size));

	//The framebuffer in device memory
	size = sizeof(unsigned int);
	CheckErrorCuda(cudaFree(dev_FrameBuffer));
	CheckErrorCuda(cudaMalloc((void**)&dev_FrameBuffer, width * height * size));
	CheckErrorCuda(cudaMemset(dev_FrameBuffer, 0, width * height * size));

	size = sizeof(int);
	CheckErrorCuda(cudaFree(dev_DepthBuffer));
	CheckErrorCuda(cudaMalloc((void**)&dev_DepthBuffer, width * height * size));
	CheckErrorCuda(cudaMemset(dev_DepthBuffer, 0, width * height * size));

	size = sizeof(int);
	cudaFree(dev_MutexBuffer);
	cudaMalloc((void**)&dev_MutexBuffer, width * height * size);
	cudaMemset(dev_MutexBuffer, 0, width * height * size);

	//NOTE: can only set data PER BYTE
	//PROBLEM: setting each byte to UCHAR_MAX (255) is impossible, since floating point numbers work differently (-nan result)
	//	0		11111110	11111111111111111111111
	//	^			^				^
	//	sign	exponent		mantissa
	//			254 - 127     2 - 2 ^ (-23)
	// 340282346638528859811704183484516925440.0   // FLT_MAX
	// 340282366920938463463374607431768211456.0   // 2^128
	//https://stackoverflow.com/questions/16350955/interpreting-the-bit-pattern-of-flt-max
	//SOLUTION:
	//Option 1: allocate float[width*height] and initialize to FLT_MAX, then memcpy (wastes lots of memory)
	//Option 2: loop through entire dev_array and set each member to FLT_MAX (too many global accesses)
	//>Option 3<: interpret depth buffer invertedly, so a depthvalue of 1.f is closest, and 0.f is furthest away from camera
	//Option 4: initialize and reset depthbuffer through additional kernel call, however this would be a lot of global memory accesses
}

CPU_CALLABLE
void CUDARenderer::AllocateMeshBuffers(const size_t numVertices, const size_t numIndices, const size_t numTriangles, unsigned int stride, size_t meshIdx)
{
	//If no sufficient space in vector, enlarge
	const size_t newSize = meshIdx + 1;
	if (newSize > dev_IVertexBuffer.size())
	{
		//TODO: reserve
		dev_IVertexBuffer.resize(newSize);
		dev_IndexBuffer.resize(newSize);
		dev_OVertexBuffer.resize(newSize);
		dev_Triangles.resize(newSize);
	}

	//Free unwanted memory
	CheckErrorCuda(cudaFree(dev_IVertexBuffer[meshIdx]));
	CheckErrorCuda(cudaFree(dev_IndexBuffer[meshIdx]));
	CheckErrorCuda(cudaFree(dev_OVertexBuffer[meshIdx]));
	CheckErrorCuda(cudaFree(dev_Triangles[meshIdx]));

	//Allocate Input Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_IVertexBuffer[meshIdx], numVertices * stride));
	//Allocate Index Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_IndexBuffer[meshIdx], numIndices * sizeof(unsigned int)));
	//Allocate Ouput Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_OVertexBuffer[meshIdx], numVertices * sizeof(OVertex)));
	//Allocate device memory for entire range of triangles
	CheckErrorCuda(cudaMalloc((void**)&dev_Triangles[meshIdx], numTriangles * sizeof(TriangleIdx)));
}

CPU_CALLABLE
void CUDARenderer::CopyMeshBuffers(float* vertexBuffer, unsigned int numVertices, short stride, unsigned int* indexBuffer, unsigned int numIndices, size_t meshIdx)
{
	//Copy Input Vertex Buffer
	CheckErrorCuda(cudaMemcpy(dev_IVertexBuffer[meshIdx], vertexBuffer, numVertices * stride, cudaMemcpyHostToDevice));
	//Copy Index Buffer
	CheckErrorCuda(cudaMemcpy(dev_IndexBuffer[meshIdx], indexBuffer, numIndices * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

CPU_CALLABLE
void CUDARenderer::LoadMeshTextures(const std::string texturePaths[4], size_t meshIdx)
{
	const size_t newSize = (meshIdx + 1);
	if (newSize > m_TextureObjects.size())
	{
		m_TextureObjects.resize(newSize);
	}
	if (newSize * NumTextures > dev_TextureData.size())
	{
		dev_TextureData.resize(newSize * NumTextures);
	}

	GPUTexturesCompact gpuTextures{};

	//0 DIFFUSE > 1 NORMAL > 2 SPECULAR > 3 GLOSSINESS
	for (int i{}; i < NumTextures; ++i)
	{
		GPUTextureCompact* gpuTexture;
		switch (i)
		{
		case 0:
			gpuTexture = &gpuTextures.Diff;
			break;
		case 1:
			gpuTexture = &gpuTextures.Norm;
			break;
		case 2:
			gpuTexture = &gpuTextures.Spec;
			break;
		case 3:
			gpuTexture = &gpuTextures.Gloss;
			break;
		}

		const unsigned int textureIdx = meshIdx * NumTextures + i;
		const GPUTexture tex = LoadGPUTexture(texturePaths[i], textureIdx);

		gpuTextures.w = tex.w;
		gpuTextures.h = tex.h;
		gpuTexture->dev_pTex = tex.dev_pTex;
		gpuTexture->dev_TextureData = tex.dev_TextureData;
	}
	//store textures
	m_TextureObjects[meshIdx] = gpuTextures;
}

CPU_CALLABLE
GPUTexture CUDARenderer::LoadGPUTexture(const std::string texturePath, unsigned int textureIdx)
{
	GPUTexture gpuTexture{};

	SDL_Surface* pSurface = IMG_Load(texturePath.c_str());
	if (pSurface)
	{
		const unsigned int width = pSurface->w;
		const unsigned int height = pSurface->h;
		const unsigned int* pixels = (unsigned int*)pSurface->pixels;
		const int bpp = pSurface->format->BytesPerPixel;
		//const size_t sizeInBytes = width * height * bpp;

		//copy texture data to device
		CheckErrorCuda(cudaFree(dev_TextureData[textureIdx]));
		size_t pitch{};
		CheckErrorCuda(cudaMallocPitch((void**)&dev_TextureData[textureIdx], &pitch, width * bpp, height)); //2D array
		CheckErrorCuda(cudaMemcpy2D(dev_TextureData[textureIdx], pitch, pixels, pitch, width * bpp, height, cudaMemcpyHostToDevice));

		//cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<unsigned int>();

		cudaResourceDesc resDesc{};
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = dev_TextureData[textureIdx];
		resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
		resDesc.res.pitch2D.desc.x = pSurface->format->BitsPerPixel;
		resDesc.res.pitch2D.width = width;
		resDesc.res.pitch2D.height = height;

		cudaTextureDesc texDesc{};
		texDesc.normalizedCoords = true; //able to sample texture with normalized uv coordinates
		texDesc.filterMode = cudaFilterModePoint; //linear only supports float (and double) type
		texDesc.readMode = cudaReadModeElementType;

		cudaTextureObject_t dev_TextureObject{};
		CheckErrorCuda(cudaCreateTextureObject(&dev_TextureObject, &resDesc, &texDesc, nullptr));

		/*{
			cudaArray* dev_array;
			cudaChannelFormatDesc formatDesc{};
			formatDesc.f = cudaChannelFormatKindUnsigned;
			formatDesc.x = 32;

			cudaMallocArray(&dev_array, &formatDesc, width, height, cudaArrayTextureGather); //2D array
			cudaMallocArray(&dev_array, &formatDesc, width * height, 1, cudaArrayTextureGather); //1D array
			cudaMemcpyToArray(dev_array, 0, 0, dev_TextureData[textureIdx], width * height * bpp, cudaMemcpyHostToDevice);

			cudaResourceDesc desc{};
			desc.resType = cudaResourceTypeArray;
			desc.res.array.array = dev_array;

			cudaTextureObject_t dev_TextureObject{};
			CheckErrorCuda(cudaCreateTextureObject(&dev_TextureObject, &desc, &texDesc, nullptr));


			cudaFreeArray(dev_array);
		}*/

		gpuTexture.dev_pTex = dev_TextureObject;
		gpuTexture.w = width;
		gpuTexture.h = height;
		gpuTexture.dev_TextureData = dev_TextureData[textureIdx];

		/*DEPRECATED
			//bind texture
			textureReference texRef{};
			texRef.normalized = false;
			texRef.channelDesc = cudaCreateChannelDesc<unsigned int>();
			texRef.channelDesc.x = bpp * 8;
			texRef.channelDesc.f = cudaChannelFormatKindUnsigned; //unsigned int

			size_t offset{};
			CUDA32bTexture2D texRef{}; //IN STATIC GLOBAL MEMORY!
			CheckErrorCuda(cudaBindTexture2D(&offset, &texRef, dev_texData2D, &texRef.channelDesc, width, height, pitch * bpp));

			if (offset != 0)
			{
				std::cout << "Texture Offset : " << offset << '\n';
				return;
			}
		*/

		//free data
		SDL_FreeSurface(pSurface);
		//!DO NOT FREE TEXTURE DATA, as this will render the texture object invalid!
	}

	return gpuTexture;
}

CPU_CALLABLE
void CUDARenderer::FreeTextures()
{
	//destroy all texture objects
	for (const GPUTexturesCompact& textures : m_TextureObjects)
	{
		CheckErrorCuda(cudaDestroyTextureObject(textures.Diff.dev_pTex));
		CheckErrorCuda(cudaDestroyTextureObject(textures.Norm.dev_pTex));
		CheckErrorCuda(cudaDestroyTextureObject(textures.Spec.dev_pTex));
		CheckErrorCuda(cudaDestroyTextureObject(textures.Gloss.dev_pTex));
	}
	m_TextureObjects.clear();
	//free texture data
	for (unsigned int* dev_texData : dev_TextureData)
	{
		CheckErrorCuda(cudaFree(dev_texData));
	}
	dev_TextureData.clear();
}

CPU_CALLABLE
void CUDARenderer::FreeMeshBuffers()
{
	for (size_t i{}; i < m_MeshIdentifiers.size(); ++i)
	{
		CheckErrorCuda(cudaFree(dev_IVertexBuffer[i]));
		dev_IVertexBuffer[i] = nullptr;
		CheckErrorCuda(cudaFree(dev_IndexBuffer[i]));
		dev_IndexBuffer[i] = nullptr;
		CheckErrorCuda(cudaFree(dev_OVertexBuffer[i]));
		dev_OVertexBuffer[i] = nullptr;
		CheckErrorCuda(cudaFree(dev_Triangles[i]));
		dev_Triangles[i] = nullptr;
	}
	m_MeshIdentifiers.clear();
	dev_IVertexBuffer.clear();
	dev_IndexBuffer.clear();
	dev_OVertexBuffer.clear();
	dev_Triangles.clear();
}

CPU_CALLABLE
void CUDARenderer::FreeCUDADeviceBuffers()
{
	//Free buffers
	CheckErrorCuda(cudaFree(dev_FrameBuffer));
	dev_FrameBuffer = nullptr;

	CheckErrorCuda(cudaFree(dev_PixelShadeBuffer));
	dev_PixelShadeBuffer = nullptr;

	CheckErrorCuda(cudaFreeHost(m_h_pFrameBuffer));
	m_h_pFrameBuffer = nullptr;

	CheckErrorCuda(cudaFree(dev_DepthBuffer));
	dev_DepthBuffer = nullptr;

	cudaFree(dev_MutexBuffer);
	dev_MutexBuffer = nullptr;

	FreeMeshBuffers();
	FreeTextures();
}

CPU_CALLABLE
void CUDARenderer::UpdateCameraDataAsync(const FPoint3& camPos, const FMatrix4& viewProjectionMatrix)
{
	//Update CamPos
	size_t numBytes = sizeof(camPos);
	CheckErrorCuda(cudaMemcpyToSymbol(dev_RenderData_const, camPos.data, numBytes, 0, cudaMemcpyHostToDevice));
	//Update ViewProjectionMatrix
	const size_t numBytesOffset = numBytes;
	numBytes = sizeof(viewProjectionMatrix);
	CheckErrorCuda(cudaMemcpyToSymbol(dev_RenderData_const, viewProjectionMatrix.data, numBytes, numBytesOffset, cudaMemcpyHostToDevice));

	//CameraDataRaw cameraData{};
	//cameraData.cameraData.camPos = camPos;
	//cameraData.cameraData.viewProjectionMatrix = viewProjectionMatrix;
	//const size_t numBytes = sizeof(CameraData);
	//void* dev_ptr{};
	//CheckErrorCuda(cudaGetSymbolAddress((void**)&dev_ptr, dev_RenderData_const));
	//CheckErrorCuda(cudaMemcpyAsync(dev_ptr, cameraData.data, numBytes, cudaMemcpyHostToDevice));
}

CPU_CALLABLE
void CUDARenderer::UpdateWorldMatrixDataAsync(const FMatrix4& worldMatrix)
{
	const size_t numBytes = sizeof(FMatrix4);
	const size_t numBytesOffset = sizeof(CameraData);
	CheckErrorCuda(cudaMemcpyToSymbolAsync(dev_RenderData_const, worldMatrix.data, numBytes, numBytesOffset, cudaMemcpyHostToDevice));

	//void* dev_ptr{};
	//CheckErrorCuda(cudaGetSymbolAddress((void**)&dev_ptr, dev_RenderData_const));
	//CheckErrorCuda(cudaMemcpyAsync(dev_ptr, worldMatrix.data, numBytes, cudaMemcpyHostToDevice));
}

CPU_CALLABLE
int CUDARenderer::EnterValidRenderingState()
{
	//https://wiki.libsdl.org/SDL_LockSurface
	int state = SDL_LockSurface(m_WindowHelper.pBackBuffer); //Set up surface for directly accessing the pixels
	//Clear screen and reset buffers
	Clear();
	return state;
}

CPU_CALLABLE
void CUDARenderer::Present()
{
	//TODO: have Vertex Shader and Rasterizer run in parallel with cudamemcpy()
	const size_t size = m_WindowHelper.Width * m_WindowHelper.Height * sizeof(unsigned int);
	CheckErrorCuda(cudaMemcpy(m_WindowHelper.pBackBufferPixels, dev_FrameBuffer, size, cudaMemcpyDeviceToHost)); //We can directly read/write from pixelbuffer
	//memcpy(m_WindowHelper.pBackBufferPixels, m_WindowHelper.h_BackBufferPixels, size);
	SDL_UnlockSurface(m_WindowHelper.pBackBuffer); //Release a surface after directly accessing the pixels.
	SDL_BlitSurface(m_WindowHelper.pBackBuffer, 0, m_WindowHelper.pFrontBuffer, 0); //Copy the window surface to the screen.
	SDL_UpdateWindowSurface(m_WindowHelper.pWindow); //Update Window's surface
}

#pragma endregion

#pragma endregion

#pragma region GPU HELPER FUNCTIONS

#pragma region ROPs

GPU_CALLABLE static
float EdgeFunction(const FPoint2& v, const FVector2& edge, const FPoint2& pixel)
{
	// clockwise
	const FVector2 vertexToPixel{ pixel - v };
	return Cross(vertexToPixel, edge);
}

GPU_CALLABLE static
bool IsPixelInTriangle(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2, const FPoint2& pixel, float weights[3])
{
	const FVector2 edgeA{ v1.xy - v0.xy };
	const FVector2 edgeB{ v2.xy - v1.xy };
	const FVector2 edgeC{ v0.xy - v2.xy };
	// clockwise
	//const FVector2 edgeA{ v0 - v1 };
	//const FVector2 edgeB{ v1 - v2 };
	//const FVector2 edgeC{ v2 - v0 };
	// counter-clockwise

	{
		//// edgeA
		//FVector2 vertexToPixel{ pixel - v0 };
		//float cross = Cross(edgeA, vertexToPixel);
		//isInTriangle &= cross < 0.f;
		//// weight2 == positive cross of 'previous' edge, for v2 this is edgeA (COUNTER-CLOCKWISE)
		//weights[2] = cross / totalArea;

		//// edgeB
		//vertexToPixel = { pixel - v1 };
		//cross = Cross(edgeB, vertexToPixel);
		//isInTriangle &= cross < 0.f;
		//// weight1 (for v1 this is edgeB)
		//weights[1] = cross / totalArea;

		//// edgeC
		//vertexToPixel = { pixel - v2 };
		//cross = Cross(edgeC, vertexToPixel);
		//isInTriangle &= cross < 0.f;
		//// weight0 (for v0 this is edgeC)
		//weights[0] = cross / totalArea;

		//weights == inverted negative cross of 'previous' edge
		//weights[0] = Cross(-vertexToPixel, edgeC) / totalArea;
		//weights[1] = Cross(-vertexToPixel, edgeB) / totalArea;
		//weights[2] = Cross(-vertexToPixel, edgeA) / totalArea;
		// gives positive results because counter-clockwise
		//const float total = weights[0] + weights[1] + weights[2]; // total result equals 1
	}

	weights[2] = EdgeFunction(v0.xy, edgeA, pixel);
	weights[0] = EdgeFunction(v1.xy, edgeB, pixel);
	weights[1] = EdgeFunction(v2.xy, edgeC, pixel);

	return weights[0] >= 0.f && weights[1] >= 0.f && weights[2] >= 0.f;
}

GPU_CALLABLE static
bool IsAllXOutsideFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return	(v0.x < -1.f && v1.x < -1.f && v2.x < -1.f) ||
		(v0.x > 1.f && v1.x > 1.f && v2.x > 1.f);
}

GPU_CALLABLE static
bool IsAllYOutsideFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return	(v0.y < -1.f && v1.y < -1.f && v2.y < -1.f) ||
		(v0.y > 1.f && v1.y > 1.f && v2.y > 1.f);
}

GPU_CALLABLE static
bool IsAllZOutsideFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return	(v0.z < 0.f && v1.z < 0.f && v2.z < 0.f) ||
		(v0.z > 1.f && v1.z > 1.f && v2.z > 1.f);
}

GPU_CALLABLE static
bool IsTriangleVisible(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	// Solution to FrustumCulling bug
	//	   if (all x values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all y values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all z values are < 0.f or > 1.f) AT ONCE, cull
	return(!IsAllXOutsideFrustum(v0, v1, v0)
		&& !IsAllYOutsideFrustum(v0, v1, v0)
		&& !IsAllZOutsideFrustum(v0, v1, v0));
}

GPU_CALLABLE static
bool IsVertexInFrustum(const FPoint4& NDC)
{
	return!((NDC.x < -1.f || NDC.x > 1.f) || 
			(NDC.y < -1.f || NDC.y > 1.f) || 
			(NDC.z < 0.f || NDC.z > 1.f));
}

GPU_CALLABLE static
bool IsTriangleInFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return(IsVertexInFrustum(v0)
		|| IsVertexInFrustum(v1)
		|| IsVertexInFrustum(v2));
	//TODO: bug, triangles gets culled when zoomed in, aka all 3 vertices are outside of frustum
}

GPU_CALLABLE static
BoundingBox GetBoundingBox(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2, const unsigned int width, const unsigned int height)
{
	BoundingBox bb;
	bb.xMin = (short)GetMinElement(v0.x, v1.x, v2.x) - 1; // xMin
	bb.yMin = (short)GetMinElement(v0.y, v1.y, v2.y) - 1; // yMin
	bb.xMax = (short)GetMaxElement(v0.x, v1.x, v2.x) + 1; // xMax
	bb.yMax = (short)GetMaxElement(v0.y, v1.y, v2.y) + 1; // yMax

	if (bb.xMin < 0) bb.xMin = 0; //clamp minX to Left of screen
	if (bb.yMin < 0) bb.yMin = 0; //clamp minY to Bottom of screen
	if (bb.xMax > width) bb.xMax = width; //clamp maxX to Right of screen
	if (bb.yMax > height) bb.yMax = height; //clamp maxY to Top of screen

	return bb;
}

GPU_CALLABLE static
void NDCToScreenSpace(FPoint4& v0, FPoint4& v1, FPoint4& v2, const unsigned int width, const unsigned int height)
{
	v0.x = ((v0.x + 1) / 2) * width;
	v0.y = ((1 - v0.y) / 2) * height;
	v1.x = ((v1.x + 1) / 2) * width;
	v1.y = ((1 - v1.y) / 2) * height;
	v2.x = ((v2.x + 1) / 2) * width;
	v2.y = ((1 - v2.y) / 2) * height;
}

#pragma endregion

#pragma region DEPRECATED ROPs

GPU_CALLABLE static
bool IsPixelInTriangle(const RasterTriangle& triangle, const FPoint2& pixel, float weights[3])
{
	return IsPixelInTriangle(triangle.v0, triangle.v1, triangle.v2, pixel, weights);
}

GPU_CALLABLE static
bool IsAllXOutsideFrustum(const RasterTriangle& triangle)
{
	return IsAllXOutsideFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsAllYOutsideFrustum(const RasterTriangle& triangle)
{
	return IsAllYOutsideFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsAllZOutsideFrustum(const RasterTriangle& triangle)
{
	return IsAllZOutsideFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsTriangleVisible(const RasterTriangle& triangle)
{
	return IsTriangleVisible(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsTriangleInFrustum(const RasterTriangle& triangle)
{
	return IsTriangleInFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
void NDCToScreenSpace(RasterTriangle& triangle, const unsigned int width, const unsigned int height)
{
	NDCToScreenSpace(triangle.v0, triangle.v1, triangle.v2, width, height);
}

GPU_CALLABLE static
BoundingBox GetBoundingBox(const RasterTriangle& triangle, const unsigned int width, const unsigned int height)
{
	return GetBoundingBox(triangle.v0, triangle.v1, triangle.v2, width, height);
}

GPU_CALLABLE static
OVertex GetNDCVertex(const IVertex& __restrict__ iVertex)
{
	const FPoint3 camPos{};
	const FMatrix4 WVPMatrix{};
	const FMatrix4 worldMatrix{};
	const FMatrix3 rotationMatrix{};

	OVertex oVertex;
	oVertex.p = WVPMatrix * FPoint4{ iVertex.p };
	oVertex.p.x /= oVertex.p.w;
	oVertex.p.y /= oVertex.p.w;
	oVertex.p.z /= oVertex.p.w;

	oVertex.n = FVector3{ rotationMatrix * iVertex.n };

	oVertex.tan = FVector3{ rotationMatrix * iVertex.tan };

	const FPoint3 worldPosition{ worldMatrix * FPoint4{ iVertex.p } };
	oVertex.vd = FVector3{ GetNormalized(worldPosition - camPos) };

	oVertex.uv = iVertex.uv;
	oVertex.c = iVertex.c;

	return oVertex;
}

#pragma endregion

GPU_CALLABLE static
bool IsPixelInBoundingBox(const FPoint2& pixel, const BoundingBox& bb)
{
	return pixel.x < bb.xMin || pixel.x > bb.xMax || pixel.y < bb.yMin || pixel.y > bb.yMax;
}

GPU_CALLABLE GPU_INLINE static
unsigned int GetStridedIdxByOffset(unsigned int globalDataIdx, unsigned int vertexStride, unsigned int valueStride, unsigned int offset = 0)
{
	//what value in row of [0, valueStride] + what vertex globally + element offset
	return (threadIdx.x % valueStride) + (globalDataIdx / valueStride) * vertexStride + offset;
}

GPU_CALLABLE static
void PerformDepthTestAtomic(int* dev_DepthBuffer, int* dev_Mutex, const unsigned int pixelIdx, float zInterpolated, PixelShade* dev_PixelShadeBuffer, const PixelShade& pixelShade)
{
	//Depth Test with correct depth interpolation
	if (zInterpolated < 0 || zInterpolated > 1.f)
		return;

	//Update depthbuffer atomically
	bool isDone = false;
	do
	{
		isDone = (atomicCAS(&dev_Mutex[pixelIdx], 0, 1) == 0);
		if (isDone)
		{
			//critical section
			if (zInterpolated < dev_DepthBuffer[pixelIdx])
			{
				dev_DepthBuffer[pixelIdx] = zInterpolated;
				dev_PixelShadeBuffer[pixelIdx] = pixelShade;
			}
			dev_Mutex[pixelIdx] = 0;
			//end of critical section
		}
	} while (!isDone);
}

GPU_CALLABLE static
void RasterizePixel(const FPoint2& pixel, const OVertex& v0, const OVertex& v1, const OVertex& v2,
	int* dev_DepthBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const GPUTexturesCompact& textures)
{
	const float v0InvDepth = 1.f / v0.p.w;
	const float v1InvDepth = 1.f / v1.p.w;
	const float v2InvDepth = 1.f / v2.p.w;

	float weights[3];
	if (IsPixelInTriangle(v0.p, v1.p, v2.p, pixel, weights))
	{
		//TODO: cull away degenerate triangles
		const float totalArea = abs(Cross(v0.p.xy - v1.p.xy, v0.p.xy - v2.p.xy));
		weights[0] /= totalArea;
		weights[1] /= totalArea;
		weights[2] /= totalArea;

		const float zInterpolated = (weights[0] * v0.p.z) + (weights[1] * v1.p.z) + (weights[2] * v2.p.z);

		//peform early depth test
		if (zInterpolated < 0 || zInterpolated > 1.f)
			return;

		const float wInterpolated = 1.f / (v0InvDepth * weights[0] + v1InvDepth * weights[1] + v2InvDepth * weights[2]);

		//create pixelshade object (== fragment)
		PixelShade pixelShade;

		//depthbuffer visualisation
		pixelShade.zInterpolated = zInterpolated;
		pixelShade.wInterpolated = wInterpolated;

		//uv
		pixelShade.uv.x = weights[0] * (v0.uv.x * v0InvDepth) + weights[1] * (v1.uv.x * v1InvDepth) + weights[2] * (v2.uv.x * v2InvDepth);
		pixelShade.uv.y = weights[0] * (v0.uv.y * v0InvDepth) + weights[1] * (v1.uv.y * v1InvDepth) + weights[2] * (v2.uv.y * v2InvDepth);
		pixelShade.uv *= wInterpolated;

		//normal
		pixelShade.n.x = weights[0] * (v0.n.x * v0InvDepth) + weights[1] * (v1.n.x * v1InvDepth) + weights[2] * (v2.n.x * v2InvDepth);
		pixelShade.n.y = weights[0] * (v0.n.y * v0InvDepth) + weights[1] * (v1.n.y * v1InvDepth) + weights[2] * (v2.n.y * v2InvDepth);
		pixelShade.n.z = weights[0] * (v0.n.z * v0InvDepth) + weights[1] * (v1.n.z * v1InvDepth) + weights[2] * (v2.n.z * v2InvDepth);
		pixelShade.n *= wInterpolated;

		//tangent
		pixelShade.tan.x = weights[0] * (v0.tan.x * v0InvDepth) + weights[1] * (v1.tan.x * v1InvDepth) + weights[2] * (v2.tan.x * v2InvDepth);
		pixelShade.tan.y = weights[0] * (v0.tan.y * v0InvDepth) + weights[1] * (v1.tan.y * v1InvDepth) + weights[2] * (v2.tan.y * v2InvDepth);
		pixelShade.tan.z = weights[0] * (v0.tan.z * v0InvDepth) + weights[1] * (v1.tan.z * v1InvDepth) + weights[2] * (v2.tan.z * v2InvDepth);

		//view direction
		pixelShade.vd.x = weights[0] * (v0.vd.x * v0InvDepth) + weights[1] * (v1.vd.x * v1InvDepth) + weights[2] * (v2.vd.x * v2InvDepth);
		pixelShade.vd.y = weights[0] * (v0.vd.y * v0InvDepth) + weights[1] * (v1.vd.y * v1InvDepth) + weights[2] * (v2.vd.y * v2InvDepth);
		pixelShade.vd.z = weights[0] * (v0.vd.z * v0InvDepth) + weights[1] * (v1.vd.z * v1InvDepth) + weights[2] * (v2.vd.z * v2InvDepth);
		Normalize(pixelShade.vd);

		//colour
		const RGBColor interpolatedColour{
			weights[0] * v0.c.r + weights[1] * v1.c.r + weights[2] * v2.c.r,
			weights[0] * v0.c.g + weights[1] * v1.c.g + weights[2] * v2.c.g,
			weights[0] * v0.c.b + weights[1] * v1.c.b + weights[2] * v2.c.b };
		pixelShade.colour = RGBA::GetRGBAFromColour(interpolatedColour).colour32;

		//store textures
		pixelShade.textures = textures;

		//multiplying z value by a INT_MAX because atomicCAS only accepts ints
		const int scaledZ = zInterpolated * INT_MAX;

		const unsigned int pixelIdx = (unsigned int)pixel.x + (unsigned int)pixel.y * width;

		//Depth Test with correct depth interpolation
		if (zInterpolated < 0 || zInterpolated > 1.f)
			return;

		if (zInterpolated < dev_DepthBuffer[pixelIdx])
		{
			dev_DepthBuffer[pixelIdx] = zInterpolated;
			dev_PixelShadeBuffer[pixelIdx] = pixelShade;
		}
	}
}

GPU_CALLABLE static
void RasterizeTriangle(const BoundingBox& bb, const OVertex& v0, const OVertex& v1, const OVertex& v2, 
	int* dev_MutexBuffer, int* dev_DepthBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const GPUTexturesCompact& textures)
{
	const float v0InvDepth = 1.f / v0.p.w;
	const float v1InvDepth = 1.f / v1.p.w;
	const float v2InvDepth = 1.f / v2.p.w;

	//Loop over all pixels in bounding box
	for (unsigned short y = bb.yMin; y < bb.yMax; ++y)
	{
		for (unsigned short x = bb.xMin; x < bb.xMax; ++x)
		{
			const FPoint2 pixel{ float(x), float(y) };
			float weights[3];
			if (IsPixelInTriangle(v0.p, v1.p, v2.p, pixel, weights))
			{
				const float totalArea = abs(Cross(v0.p.xy - v1.p.xy, v0.p.xy - v2.p.xy));
				weights[0] /= totalArea;
				weights[1] /= totalArea;
				weights[2] /= totalArea;

				const size_t pixelIdx = x + y * width;
				const float zInterpolated = (weights[0] * v0.p.z) + (weights[1] * v1.p.z) + (weights[2] * v2.p.z);

				//peform early depth test
				if (zInterpolated < 0 || zInterpolated > 1.f)
					continue;

				const float wInterpolated = 1.f / (v0InvDepth * weights[0] + v1InvDepth * weights[1] + v2InvDepth * weights[2]);

				//create pixelshade object (== fragment)
				PixelShade pixelShade;

				//depthbuffer visualisation
				pixelShade.zInterpolated = zInterpolated;
				pixelShade.wInterpolated = wInterpolated;

				//uv
				pixelShade.uv.x = weights[0] * (v0.uv.x * v0InvDepth) + weights[1] * (v1.uv.x * v1InvDepth) + weights[2] * (v2.uv.x * v2InvDepth);
				pixelShade.uv.y = weights[0] * (v0.uv.y * v0InvDepth) + weights[1] * (v1.uv.y * v1InvDepth) + weights[2] * (v2.uv.y * v2InvDepth);
				pixelShade.uv *= wInterpolated;

				//normal
				pixelShade.n.x = weights[0] * (v0.n.x * v0InvDepth) + weights[1] * (v1.n.x * v1InvDepth) + weights[2] * (v2.n.x * v2InvDepth);
				pixelShade.n.y = weights[0] * (v0.n.y * v0InvDepth) + weights[1] * (v1.n.y * v1InvDepth) + weights[2] * (v2.n.y * v2InvDepth);
				pixelShade.n.z = weights[0] * (v0.n.z * v0InvDepth) + weights[1] * (v1.n.z * v1InvDepth) + weights[2] * (v2.n.z * v2InvDepth);
				pixelShade.n *= wInterpolated;

				//tangent
				pixelShade.tan.x = weights[0] * (v0.tan.x * v0InvDepth) + weights[1] * (v1.tan.x * v1InvDepth) + weights[2] * (v2.tan.x * v2InvDepth);
				pixelShade.tan.y = weights[0] * (v0.tan.y * v0InvDepth) + weights[1] * (v1.tan.y * v1InvDepth) + weights[2] * (v2.tan.y * v2InvDepth);
				pixelShade.tan.z = weights[0] * (v0.tan.z * v0InvDepth) + weights[1] * (v1.tan.z * v1InvDepth) + weights[2] * (v2.tan.z * v2InvDepth);

				//view direction
				pixelShade.vd.x = weights[0] * (v0.vd.x * v0InvDepth) + weights[1] * (v1.vd.x * v1InvDepth) + weights[2] * (v2.vd.x * v2InvDepth);
				pixelShade.vd.y = weights[0] * (v0.vd.y * v0InvDepth) + weights[1] * (v1.vd.y * v1InvDepth) + weights[2] * (v2.vd.y * v2InvDepth);
				pixelShade.vd.z = weights[0] * (v0.vd.z * v0InvDepth) + weights[1] * (v1.vd.z * v1InvDepth) + weights[2] * (v2.vd.z * v2InvDepth);
				Normalize(pixelShade.vd);

				//colour
				const RGBColor interpolatedColour{
					weights[0] * v0.c.r + weights[1] * v1.c.r + weights[2] * v2.c.r,
					weights[0] * v0.c.g + weights[1] * v1.c.g + weights[2] * v2.c.g,
					weights[0] * v0.c.b + weights[1] * v1.c.b + weights[2] * v2.c.b };
				pixelShade.colour = RGBA::GetRGBAFromColour(interpolatedColour).colour32;

				//store textures
				pixelShade.textures = textures;

				//multiplying z value by a INT_MAX because atomicCAS only accepts ints
				const int scaledZ = zInterpolated * INT_MAX;

				PerformDepthTestAtomic(dev_DepthBuffer, dev_MutexBuffer, pixelIdx, scaledZ, dev_PixelShadeBuffer, pixelShade);
			}
		}
	}
}

GPU_CALLABLE GPU_INLINE static
RGBColor ShadePixel(const GPUTexturesCompact& textures, const FVector2& uv, const FVector3& n, const FVector3& tan, const FVector3& vd, 
	SampleState sampleState, bool isFlipGreenChannel = false)
{
	RGBColor finalColour{};

	//global settings
	const RGBColor ambientColour{ 0.05f, 0.05f, 0.05f };
	const FVector3 lightDirection = { 0.577f, -0.577f, -0.577f };
	const float lightIntensity = 7.0f;

	// texture sampling
	const RGBColor diffuseSample = GPUTextureSampler::Sample(textures.Diff, textures.w, textures.h, uv, sampleState);

	if (textures.Norm.dev_pTex != 0)
	{
		const RGBColor normalSample = GPUTextureSampler::Sample(textures.Norm, textures.w, textures.h, uv, sampleState);

		// normal mapping
		FVector3 binormal = Cross(tan, n);
		if (isFlipGreenChannel)
			binormal = -binormal;
		const FMatrix3 tangentSpaceAxis{ tan, binormal, n };

		FVector3 finalNormal{ 2.f * normalSample.r - 1.f, 2.f * normalSample.g - 1.f, 2.f * normalSample.b - 1.f };
		finalNormal = tangentSpaceAxis * finalNormal;

		// light calculations
		float observedArea{ Dot(-finalNormal, lightDirection) };
		Clamp(observedArea, 0.f, observedArea);
		observedArea /= (float)PI;
		observedArea *= lightIntensity;
		const RGBColor diffuseColour = diffuseSample * observedArea;

		if (textures.Spec.dev_pTex != 0 && textures.Gloss.dev_pTex != 0)
		{
			const RGBColor specularSample = GPUTextureSampler::Sample(textures.Spec, textures.w, textures.h, uv, sampleState);
			const RGBColor glossSample = GPUTextureSampler::Sample(textures.Gloss, textures.w, textures.h, uv, sampleState);

			// phong specular
			const FVector3 reflectV{ Reflect(lightDirection, finalNormal) };
			float angle{ Dot(reflectV, vd) };
			Clamp(angle, 0.f, 1.f);
			const float shininess = 25.f;
			angle = powf(angle, glossSample.r * shininess);
			const RGBColor specularColour = specularSample * angle;

			// final
			finalColour = ambientColour + diffuseColour + specularColour;
			finalColour.ClampColor();
		}
		else
		{
			finalColour = diffuseColour;
		}
	}
	else
	{
		finalColour = diffuseSample;
	}
	return finalColour;
}

GPU_CALLABLE GPU_INLINE static
void MultiplyMatVec(const float* pMat, float* pVec, unsigned int matSize, unsigned int vecSize)
{
	//thread goes through each element of vector
	float vec[4]{};
	for (unsigned int element{}; element < vecSize; ++element)
	{
		float sum{};
		for (unsigned int i{}; i < matSize; ++i)
		{
			sum += pMat[(element * matSize) + i] * pVec[i];
		}
		vec[element] = sum;
	}
	memcpy(pVec, vec, vecSize * 4);
}

GPU_CALLABLE GPU_INLINE static
void CalculateOutputPosXYZ(const float* pMat, float* pVec, float* pW)
{
	constexpr unsigned int matSize = 4;
	constexpr unsigned int vecSize = 3;

	//thread goes through each element of vector
	float vec[4]{};
	for (unsigned int element{}; element < vecSize; ++element)
	{
		for (unsigned int i{}; i < vecSize; ++i)
		{
			vec[element] += pMat[(element * matSize) + i] * pVec[i];
		}
		vec[element] += pMat[(element * matSize) + 3]; // * pVec[w] == 1.f
	}

	for (unsigned int i{}; i < vecSize; ++i)
	{
		vec[3] += pMat[12 + i] * pVec[i];
	}
	vec[3] += pMat[15]; // * pVec[w] == 1.f

	memcpy(pVec, vec, 12);
	*pW = vec[3];
}

#pragma endregion

#pragma region KERNELS
//Kernel launch params:	numBlocks, numThreadsPerBlock, numSharedMemoryBytes, stream

#pragma region Clearing

GPU_KERNEL
void ClearDepthBufferKernel(int* dev_DepthBuffer, int value, const unsigned int width, const unsigned int height)
{
	//loading into shared memory and then copying to global memory proves to be slightly faster in most cases
	GPU_SHARED_MEMORY int a[1];
	a[0] = value;

	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_DepthBuffer[pixelIdx] = a[0];
	}
}

GPU_KERNEL
void ClearFrameBufferKernel(unsigned int* dev_FrameBuffer, const unsigned int width, const unsigned int height, unsigned int colour32)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_FrameBuffer[pixelIdx] = colour32;
	}
}

GPU_KERNEL
void ClearPixelShadeBufferKernel(PixelShade* dev_PixelShadeBuffer, const unsigned int sizeInWords)
{
	//every thread sets 1 WORD of data
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < sizeInWords)
	{
		reinterpret_cast<float*>(dev_PixelShadeBuffer)[idx] = 0.f;
	}
}

GPU_KERNEL
void ClearMutexBufferKernel(int* dev_MutexBuffer, const unsigned int sizeInWords, int value)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < sizeInWords)
	{
		dev_MutexBuffer[idx] = value;
	}
}

GPU_KERNEL
void Clear()
{
	//TODO: clear depthbuffer, framebuffer and pixelshadebuffer
}

#pragma endregion

GPU_KERNEL
void VertexShaderKernel(const IVertex* __restrict__ dev_IVertices, OVertex* dev_OVertices, unsigned int numVertices)
{
	//TODO: store matrix in top of shared memory for faster access
	//and offset shared memory access for threads
	//Potential problem: first warp might encounter bank conflicts?

	//OVERVIEW: each thread manages 1 attribute, this being a Vector3
	//The Output Position is being stored as a Vector3, with the W-elements stored in a separate shared memory row
	//32 x 3 = 96 => 32 vertex attributes per warp in a shared memory buffer of size 96
	//32 x 4 = 128 => 32 vertex OPositions per warp in a shared memory buffer of size 128

	//IVERTEX LAYOUT: POS3 - UV2 - NORM3 - TAN3 - COL3 (size: 14)
	//OVERTEX LAYOUT: POS4 - UV2 - NORM3 - TAN3 - VD3 - COL3 (size: 18)

	GPU_SHARED_MEMORY float sharedMemoryBuffer[128];

	const float* WVPMatrix = reinterpret_cast<const float*>(dev_WVPMatrix_const);
	const float* worldMatrix = reinterpret_cast<const float*>(dev_WorldMatrix_const);
	const float* rotationMatrix = reinterpret_cast<const float*>(dev_RotationMatrix_const);
	const float* camPos = reinterpret_cast<const float*>(dev_CameraPos_const);

	const unsigned int vertexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	//for every 32 threads of vec3 (96 elements), a row of W elements is created (32 elements)

	//TODO: each thread should store 1 bank element at once for more coalesced access
	//instead of 1 thread storing 1 attribute from multiple banks to global memory

	if (vertexIdx < numVertices)
	{

		// --- STEP 1 ---: Calculate Input Position to Ouput Position

		{
			//strided load into shared memory
			const unsigned int sharedMemVecIdx = threadIdx.x * 3 + (threadIdx.x / 32) * 32;
			float* pPosXYZ = &sharedMemoryBuffer[sharedMemVecIdx];
			const unsigned int sharedMemWIdx = threadIdx.x + (threadIdx.x / 32) * 96;
			float* pPosW = &sharedMemoryBuffer[sharedMemWIdx];

			memcpy(pPosXYZ, &dev_IVertices[vertexIdx].p, 12);
			//memory is now coalesced
			__syncthreads();

			//calculate NDC (WVP * v.p.xyzw / w)
			CalculateOutputPosXYZ(WVPMatrix, pPosXYZ, pPosW); //calculate NDC (WVPMat)
			__syncthreads();

			//divide xyz by w
			pPosXYZ[0] /= *pPosW;
			pPosXYZ[1] /= *pPosW;
			pPosXYZ[2] /= *pPosW;

			//store into global memory
			OVertex* pOVertex = &dev_OVertices[vertexIdx];
			memcpy(&pOVertex->p, pPosXYZ, 12); //copy vec3 elements
			pOVertex->p.w = *pPosW; //copy w element
		}

		// --- STEP 2 ---: Calculate Input Normal to Output Normal

		{
			float* pNormal = &sharedMemoryBuffer[threadIdx.x * 3];
			memcpy(pNormal, &dev_IVertices[vertexIdx].n, 12);
			__syncthreads();

			MultiplyMatVec(rotationMatrix, pNormal, 3, 3); //calculate normal
			__syncthreads();

			OVertex* pOVertex = &dev_OVertices[vertexIdx];
			memcpy(&pOVertex->n, pNormal, 12);
		}

		// --- STEP 3 ---: Calculate Input Tangent to Output Tangent

		{
			float* pTan = &sharedMemoryBuffer[threadIdx.x * 3];
			memcpy(pTan, &dev_IVertices[vertexIdx].tan, 12);
			__syncthreads();

			MultiplyMatVec(rotationMatrix, pTan, 3, 3); //calculate tangent
			__syncthreads();

			OVertex* pOVertex = &dev_OVertices[vertexIdx];
			memcpy(&pOVertex->tan, pTan, 12);
		}

		{
			// --- STEP 4 ---: Calculate ViewDirection

			const unsigned int sharedMemVecIdx = threadIdx.x * 3 + (threadIdx.x / 32) * 32;
			float* pVecXYZ = &sharedMemoryBuffer[sharedMemVecIdx];
			const unsigned int sharedMemWIdx = threadIdx.x + (threadIdx.x / 32) * 96;
			float* pVecW = &sharedMemoryBuffer[sharedMemWIdx];

			memcpy(pVecXYZ, &dev_IVertices[vertexIdx].p, 12);
			__syncthreads();

			CalculateOutputPosXYZ(worldMatrix, pVecXYZ, pVecW); //calculate worldposition (worldMat)
			__syncthreads();

			pVecXYZ[0] -= camPos[0];
			pVecXYZ[1] -= camPos[1];
			pVecXYZ[2] -= camPos[2];

			Normalize(reinterpret_cast<FVector3&>(pVecXYZ));

			OVertex* pOVertex = &dev_OVertices[vertexIdx];
			memcpy(&pOVertex->vd, pVecXYZ, 12);
		}

		// --- STEP 5 ---: Copy UV and Colour

		{
			//COLOUR
			float* pCol = &sharedMemoryBuffer[threadIdx.x * 3];
			memcpy(pCol, &dev_IVertices[vertexIdx].c, 12);
			__syncthreads();

			memcpy(&dev_OVertices[vertexIdx].c, pCol, 12);
		}

		{

			//UV
			float* pUV = &sharedMemoryBuffer[threadIdx.x * 3]; //"padded" to avoid bank conflicts
			memcpy(pUV, &dev_IVertices[vertexIdx].uv, 8);
			__syncthreads();

			memcpy(&dev_OVertices[vertexIdx].uv, pUV, 8);
		}
	}
}

GPU_KERNEL
void TriangleAssemblerKernel(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer, unsigned int numIndices, 
	OVertex* dev_OVertices, const PrimitiveTopology pt)
{
	//TODO: perform culling/clipping etc.
	//advantage of TriangleAssembly: each thread stores 1 triangle
	//many threads == many triangles processed at once

	//'talk about naming gore, eh?
	const unsigned int indexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pt == PrimitiveTopology::TriangleList)
	{
		const unsigned int correctedIdx = (indexIdx * 3);
		if (correctedIdx < numIndices)
		{
			//Unnecessary for TriangleLists
			//TriangleIdx triangle;
			//triangle.idx0 = dev_IndexBuffer[correctedIdx];
			//triangle.idx1 = dev_IndexBuffer[correctedIdx + 1];
			//triangle.idx2 = dev_IndexBuffer[correctedIdx + 2];
			//triangle.isCulled = false;
			//dev_Triangles[indexIdx] = triangle;

			memcpy(&dev_Triangles[indexIdx], &dev_IndexBuffer[correctedIdx], sizeof(TriangleIdx));
			//atomically increment visible triangle count
			//atomicAdd(dev_VisibleNumTriangles, 1);
		}
	}
	else //if (pt == PrimitiveTopology::TriangleStrip)
	{
		if (indexIdx < numIndices - 2)
		{
			//Necessary for TriangleStrips
			TriangleIdx triangle;
			const bool isOdd = (indexIdx % 2);
			//triangle.idx0 = dev_IndexBuffer[indexIdx];
			//if (isOdd)
			//{
			//	triangle.idx1 = dev_IndexBuffer[indexIdx + 2];
			//	triangle.idx2 = dev_IndexBuffer[indexIdx + 1];
			//}
			//else
			//{
			//	triangle.idx1 = dev_IndexBuffer[indexIdx + 1];
			//	triangle.idx2 = dev_IndexBuffer[indexIdx + 2];
			//}
			//triangle.isCulled = false;

			memcpy(&triangle, &dev_IndexBuffer[indexIdx], sizeof(TriangleIdx));
			if (isOdd)
			{
				const unsigned int origIdx1 = triangle.idx1;
				triangle.idx1 = triangle.idx2;
				triangle.idx2 = origIdx1;
			}
			dev_Triangles[indexIdx] = triangle;

			//atomically increment visible triangle count
			//atomicAdd(dev_VisibleNumTriangles, 1);
		}
	}
}

GPU_KERNEL
void RasterizerKernelCTA(const TriangleIdx* __restrict__ dev_Triangles, const OVertex* __restrict__ const dev_OVertices, unsigned int numTriangles,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, GPUTexturesCompact textures, 
	const unsigned int width, const unsigned int height)
{
	//TODO: each thread represents a pixel
	//each thread loops through all triangles
	//triangles are stored in shared memory (broadcast)
	//advantage: thread only does 1 check per triangle w/o looping for all pixels 
	//=> O(n) n = numTriangles vs O(n^m) n = numTriangles m = numPixels
	//advantage: nomore atomic operations needed bc only 1 thread can write to 1 unique pixelIdx

	GPU_SHARED_MEMORY float sharedMemoryBuffer[32];

	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	const FPoint2 pixel{ float(x), float(y) };

	for (unsigned int i{}; i < numTriangles; ++i)
	{
		const TriangleIdx triangleIdx = dev_Triangles[i];

		//TODO: store in shared memory
		OVertex v0 = dev_OVertices[triangleIdx.idx0];
		OVertex v1 = dev_OVertices[triangleIdx.idx1];
		OVertex v2 = dev_OVertices[triangleIdx.idx2];

		if (!IsTriangleVisible(v0.p, v1.p, v2.p))
		{
			return;
		}

		NDCToScreenSpace(v0.p, v1.p, v2.p, width, height);
		const BoundingBox bb = GetBoundingBox(v0.p, v1.p, v2.p, width, height);

		if (!IsPixelInBoundingBox(pixel, bb))
		{
			return;
		}

		//Rasterize pixel
		RasterizePixel(pixel, v0, v1, v2, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);
	}
}

GPU_KERNEL
void RasterizerKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices, unsigned int numTriangles,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, int* dev_MutexBuffer, GPUTexturesCompact textures,
	const FVector3 camFwd, const CullingMode cm, const unsigned int width, const unsigned int height)
{
	//Every thread processes 1 single triangle for now
	const unsigned int globalTriangleIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (!(globalTriangleIndex < numTriangles))
		return;

	const TriangleIdx triangleIdx = dev_Triangles[globalTriangleIndex];

	OVertex v0 = dev_OVertices[triangleIdx.idx0];
	OVertex v1 = dev_OVertices[triangleIdx.idx1];
	OVertex v2 = dev_OVertices[triangleIdx.idx2];

	bool isDoubleSidedRendering = false;

	//is triangle visible according to cullingmode?
	if (cm == CullingMode::BackFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue <= 0.f)
		{
			if (isDoubleSidedRendering)
			{
				OVertex origV1 = v1;
				v1 = v2;
				v2 = origV1;
			}
			else
			{
				return; //cull triangle
			}
		}
	}
	else if (cm == CullingMode::FrontFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue >= 0.f)
		{
			if (isDoubleSidedRendering)
			{
				OVertex origV1 = v1;
				v1 = v2;
				v2 = origV1;
			}
			else
			{
				return; //cull triangle
			}
		}
	}
	//else if (cm == CullingMode::NoCulling)
	//{
	//}

	if (!IsTriangleVisible(v0.p, v1.p, v2.p))
	{
		return;
	}

	NDCToScreenSpace(v0.p, v1.p, v2.p, width, height);
	const BoundingBox bb = GetBoundingBox(v0.p, v1.p, v2.p, width, height);
	//Rasterize Screenspace triangle
	RasterizeTriangle(bb, v0, v1, v2, dev_MutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);
}

GPU_KERNEL
void PixelShaderKernel(unsigned int* dev_FrameBuffer, const PixelShade* __restrict__ const dev_PixelShadeBuffer,
	SampleState sampleState, bool isDepthColour, const unsigned int width, const unsigned int height)
{
	//Notes: PixelShade has size of 32, but bank conflicts

	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		RGBA rgba;
		if (isDepthColour)
		{
			rgba.r8 = (unsigned char)(Remap(dev_PixelShadeBuffer[pixelIdx].zInterpolated, 0.985f, 1.f) * 255);
			rgba.g8 = 0;
			rgba.b8 = 0;
			rgba.a8 = 0;
			dev_FrameBuffer[pixelIdx] = rgba.colour32;
		}
		else
		{
			const PixelShade pixelShade = dev_PixelShadeBuffer[pixelIdx];
			if (pixelShade.textures.Diff.dev_pTex != 0)
			{
				RGBColor colour = ShadePixel(pixelShade.textures, pixelShade.uv, pixelShade.n, pixelShade.tan, pixelShade.vd, sampleState);
				rgba = colour; //== GetRGBAFromColour()
				dev_FrameBuffer[pixelIdx] = rgba.colour32;
			}
			else
			{
				dev_FrameBuffer[pixelIdx] = pixelShade.colour;
			}
		}
	}
}

GPU_KERNEL
void TextureTestKernel(unsigned int* dev_FrameBuffer, GPUTexture texture, const unsigned int width, const unsigned int height)
{
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		float u = float(x) / width;
		float v = float(y) / height;
		//u *= texture.w;
		//v *= texture.h;
		//float uC = Clamp(float(x), 0.f, (float)texture.w);
		//float vC = Clamp(float(y), 0.f, (float)texture.h);
		//float sampleIdx = u + v * texture.w;
		//unsigned int sample = tex1Dfetch<unsigned int>(texture.dev_pTex, (int)sampleIdx);
		//remap uv's to stretch towards the window's dimensions
		unsigned int sample = tex2D<unsigned int>(texture.dev_pTex, u, v);
		RGBA rgba = sample;
		unsigned char b = rgba.b8;
		rgba.b8 = rgba.r8;
		rgba.r8 = b;
		dev_FrameBuffer[pixelIdx] = rgba.colour32;
	}
}

GPU_KERNEL
void DrawTextureGlobalKernel(unsigned int* dev_FrameBuffer, GPUTexture texture, bool isStretchedToWindow,
	SampleState sampleState, const unsigned int width, const unsigned int height)
{
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		//remap uv's to stretch towards the window's dimensions
		FVector2 uv;
		uv.x = float(x);
		uv.y = float(y);
		if (isStretchedToWindow)
		{
			uv.x /= width;
			uv.y /= height;
		}
		else
		{
			uv.x /= texture.w;
			uv.y /= texture.h;
		}
		RGBColor sample = GPUTextureSampler::Sample(texture, uv, sampleState);
		RGBA rgba = sample;
		dev_FrameBuffer[pixelIdx] = rgba.colour32;
	}
}

#pragma endregion

#pragma region KERNEL LAUNCHERS

CPU_CALLABLE
void CUDARenderer::Clear(const RGBColor& colour)
{
	{
		const int depthBufferResetValue = INT_MAX;
		const dim3 numThreadsPerBlock{ 16, 16 };
		const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
		ClearDepthBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_DepthBuffer, depthBufferResetValue, m_WindowHelper.Width, m_WindowHelper.Height);
	}

	{
		const RGBA rgba{ colour };
		const size_t sizeInWords = m_WindowHelper.Width * m_WindowHelper.Height;
		const unsigned int numThreadsPerBlock = 512;
		const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
		ClearFrameBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_FrameBuffer, m_WindowHelper.Width, m_WindowHelper.Height, rgba.colour32);
	}

	{
		const size_t sizeInWords = m_WindowHelper.Width * m_WindowHelper.Height * (sizeof(PixelShade) / 4);
		const unsigned int numThreadsPerBlock = 1024;
		const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
		ClearPixelShadeBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_PixelShadeBuffer, sizeInWords);
	}

	{
		const size_t sizeInWords = m_WindowHelper.Width * m_WindowHelper.Height * (sizeof(PixelShade) / 4);
		const unsigned int numThreadsPerBlock = 1024;
		const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
		ClearMutexBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_MutexBuffer, sizeInWords, INT_MAX);
	}
}

CPU_CALLABLE
void CUDARenderer::VertexShader(const MeshIdentifier& mi)
{
	//TODO: define size of dynamic shared memory by checking whether 
	//		amount of vertices >= 65535 (max number of blocks) / amount of vertices that are able to be processed in 1 CTA/block (8 here)
	//		this way we can have CTAs process double the amount of vertices without bank conflicts (slower, but good scaling method (for now))

	const unsigned int numVertices = mi.pMesh->GetVertexAmount();
	const unsigned int numThreadsPerBlock = 32;
	//launch kernel with 8 vertices / block
	const unsigned int numBlocks = ((numVertices * 4) + numThreadsPerBlock - 1) / numThreadsPerBlock;
	const IVertex* pIVertices = dev_IVertexBuffer[mi.Idx];
	OVertex* pOVertices = dev_OVertexBuffer[mi.Idx];
	VertexShaderKernel<<<numBlocks, numThreadsPerBlock>>>(
		pIVertices, pOVertices, numVertices);
}

CPU_CALLABLE
void CUDARenderer::TriangleAssembler(MeshIdentifier& mi)
{
	//unsigned int* dev_VisibleNumTriangles;
	//CheckErrorCuda(cudaMalloc((void**)&dev_VisibleNumTriangles, sizeof(unsigned int)));

	const unsigned int numIndices = mi.pMesh->GetIndexAmount();
	const PrimitiveTopology topology = mi.pMesh->GetTopology();

	const unsigned int numThreadsPerBlock = 256;
	unsigned int numBlocks = (numIndices + numThreadsPerBlock - 1) / numThreadsPerBlock;
	if (topology == PrimitiveTopology::TriangleList)
		numBlocks = ((numIndices / 3) + numThreadsPerBlock - 1) / numThreadsPerBlock;
	//OCCUPANCY does strange things (tiny performance loss on current setup)
	//For TriangleStrips it would waste 2 threads at max, not too big of an issue here
	//else if (topology == PrimitiveTopology::TriangleStrip)
		//numBlocks = ((unsigned int)numIndices - 2 + numThreadsPerBlock - 1) / numThreadsPerBlock;
	TriangleAssemblerKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_Triangles[mi.Idx], dev_IndexBuffer[mi.Idx], numIndices, 
		dev_OVertexBuffer[mi.Idx], topology);

	//CheckErrorCuda(cudaDeviceSynchronize());
	//CheckErrorCuda(cudaMemcpy(&mi.VisibleNumTriangles, dev_VisibleNumTriangles, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//m_TotalVisibleNumTriangles += mi.VisibleNumTriangles;

	//CheckErrorCuda(cudaFree(dev_VisibleNumTriangles));
}

CPU_CALLABLE
void CUDARenderer::Rasterizer(const MeshIdentifier& mi, const FVector3& camFwd, const CullingMode cm)
{
	const dim3 numThreadsPerBlock = { 32, 32 };
	const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	//RasterizerKernel<<<numBlocks, numThreadsPerBlock>>>(
	//	dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], mi.TotalNumTriangles,
	//	dev_PixelShadeBuffer, dev_DepthBuffer, dev_MutexBuffer, mi.Textures,
	//	camFwd, cm, m_WindowHelper.Width, m_WindowHelper.Height);

	RasterizerKernelCTA<<<numBlocks, numThreadsPerBlock>>>(
		dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], mi.TotalNumTriangles,
		dev_PixelShadeBuffer, dev_DepthBuffer, mi.Textures,
		m_WindowHelper.Width, m_WindowHelper.Height);
}

CPU_CALLABLE
void CUDARenderer::PixelShader(SampleState sampleState, bool isDepthColour)
{
	const dim3 numThreadsPerBlock{ 32, 32 };
	const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	PixelShaderKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_FrameBuffer, dev_PixelShadeBuffer, sampleState, isDepthColour,
		m_WindowHelper.Width, m_WindowHelper.Height);
}

CPU_CALLABLE
void CUDARenderer::DrawTexture(char* tP)
{
	SDL_Surface* pS = IMG_Load(tP);

	int w = pS->w;
	int h = pS->h;
	int bpp = pS->format->BytesPerPixel;
	unsigned int* buffer;
	size_t pitch{};
	CheckErrorCuda(cudaMallocPitch((void**)&buffer, &pitch, w * bpp, h)); //2D array
	CheckErrorCuda(cudaMemcpy2D(buffer, pitch, buffer, pitch, w * bpp, h, cudaMemcpyHostToDevice));

	//cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<unsigned int>();

	cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = buffer;
	resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.pitch2D.desc.x = pS->format->BitsPerPixel;
	resDesc.res.pitch2D.width = w;
	resDesc.res.pitch2D.height = h;
	resDesc.res.pitch2D.pitchInBytes = pitch;

	cudaTextureDesc texDesc{};
	texDesc.normalizedCoords = true; //able to sample texture with normalized uv coordinates
	texDesc.filterMode = cudaFilterModePoint; //linear only supports float (and double) type
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t tex{};
	CheckErrorCuda(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

	GPUTexture texture{};
	texture.dev_pTex = tex;
	texture.w = w;
	texture.h = h;
	texture.dev_TextureData = buffer;

	EnterValidRenderingState();

	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	TextureTestKernel<<<numBlocks, numThreadsPerBlock>>>(dev_FrameBuffer, texture, m_WindowHelper.Width, m_WindowHelper.Height);

	Present();

	//destroy texture object
	CheckErrorCuda(cudaDestroyTextureObject(tex));

	SDL_FreeSurface(pS);

	//do not free buffer if it is meant to be reused
	CheckErrorCuda(cudaFree(buffer));
}

CPU_CALLABLE
void CUDARenderer::DrawTextureGlobal(char* tp, bool isStretchedToWindow, SampleState sampleState)
{
	SDL_Surface* pS = IMG_Load(tp);

	int w = pS->w;
	int h = pS->h;
	int N = w * h;
	unsigned int* buffer;
	cudaMalloc(&buffer, N * sizeof(unsigned int));
	cudaMemcpy(buffer, pS->pixels, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

	EnterValidRenderingState();

	GPUTexture gpuTexture{};
	gpuTexture.dev_pTex = 0; //none
	gpuTexture.dev_TextureData = buffer;
	gpuTexture.w = w;
	gpuTexture.h = h;

	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	DrawTextureGlobalKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_FrameBuffer, gpuTexture, isStretchedToWindow, 
		sampleState, m_WindowHelper.Width, m_WindowHelper.Height);

	Present();

	SDL_FreeSurface(pS);

	cudaFree(buffer);
}

CPU_CALLABLE
void CUDARenderer::WarmUp()
{
	ClearDepthBufferKernel<<<0, 0>>>(nullptr, 0, 0, 0);
	ClearFrameBufferKernel<<<0, 0>>>(nullptr, 0, 0, 0);
	ClearPixelShadeBufferKernel<<<0, 0>>>(nullptr, 0);
	ClearMutexBufferKernel<<<0, 0>>>(nullptr, 0, 0);
	VertexShaderKernel<<<0, 0>>>(nullptr, nullptr, 0);
	TriangleAssemblerKernel<<<0, 0>>>(nullptr, nullptr, 0, nullptr, (PrimitiveTopology)0);
	RasterizerKernel<<<0, 0>>>(nullptr, nullptr, 0, nullptr, nullptr, nullptr, {}, {}, (CullingMode)0, 0, 0);
	PixelShaderKernel<<<0, 0>>>(nullptr, nullptr, SampleState(0), false, 0, 0);
}

#pragma endregion