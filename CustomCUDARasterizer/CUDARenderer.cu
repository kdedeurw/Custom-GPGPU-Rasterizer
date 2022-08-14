#include "PCH.h"
#include "CUDARenderer.h"
#include <vector>

//Project CUDA includes
#include "GPUTextureSampler.cuh"
#include "CUDAMatrixMath.cuh"
#include "RasterizerOperations.cu"

#pragma region STRUCT DECLARATIONS

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

//CONST DEVICE MEMORY - Does NOT have to be allocated or freed
constexpr unsigned int ConstMemorySize = 256;
GPU_CONST_MEMORY float dev_ConstMemory[ConstMemorySize];
//GPU_CONST_MEMORY float dev_CameraPos_const[sizeof(FPoint3) / sizeof(float)];
//GPU_CONST_MEMORY float dev_WVPMatrix_const[sizeof(FMatrix4) / sizeof(float)];
//GPU_CONST_MEMORY float dev_WorldMatrix_const[sizeof(FMatrix4) / sizeof(float)];
//GPU_CONST_MEMORY float dev_RotationMatrix_const[sizeof(FMatrix3) / sizeof(float)];

constexpr int NumTextures = 4;

//DEVICE MEMORY - Does have to be allocated and freed
static unsigned int* dev_FrameBuffer{};
static int* dev_DepthBuffer{}; //defined as INTEGER type for atomicCAS to work properly
static int* dev_MutexBuffer{};
static PixelShade* dev_PixelShadeBuffer{}; //(== fragmentbuffer)
static std::vector<IVertex*> dev_IVertexBuffer{};
static std::vector<unsigned int*> dev_IndexBuffer{};
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
	, m_h_pFrameBuffer{}
	, m_MeshIdentifiers{}
	, m_TextureObjects{}
	, m_BenchMarker{}
{
	InitCUDADeviceBuffers();
}

CPU_CALLABLE
CUDARenderer::~CUDARenderer()
{
	CheckErrorCuda(DeviceSynchroniseCuda());
	FreeCUDADeviceBuffers();
}

#pragma region CPU HELPER FUNCTIONS

#pragma region PUBLIC FUNCTIONS

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
	std::cout << "Max threads Dimensions: X: " << prop.maxThreadsDim[0] << ", Y: " << prop.maxThreadsDim[1] << ", Z: " << prop.maxThreadsDim[2] << '\n';
	std::cout << "Warp Size (in threads): " << prop.warpSize << '\n';
	std::cout << '\n';
}

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

		const std::vector<IVertex> vertexBuffer = pMesh->GetVertexBuffer();
		const std::vector<unsigned int> indexBuffer = pMesh->GetIndexBuffer();
		const unsigned int numVertices = pMesh->GetVertexAmount();
		const unsigned int numIndices = pMesh->GetIndexAmount();
		const PrimitiveTopology topology = pMesh->GetTopology();
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

		AllocateMeshBuffers(numVertices, numIndices, numTriangles, stride, mi.Idx);
		const float* pVertices = reinterpret_cast<const float*>(vertexBuffer.data());
		const unsigned int* pIndices = reinterpret_cast<const unsigned int*>(indexBuffer.data());
		CopyMeshBuffers(pVertices, numVertices, stride, pIndices, numIndices, mi.Idx);
		if (!pMesh->GetTexPaths()->empty())
		{
			GPUTexturesCompact gpuTextures = LoadMeshTextures(pMesh->GetTexPaths(), mi.Idx);
			m_TextureObjects[mi.Idx] = gpuTextures;
			mi.Textures = gpuTextures;
		}

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

	UpdateCameraDataAsync(camPos, camFwd);

	//SceneGraph Data
	const SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pMeshes = pSceneGraph->GetMeshes();

#ifdef BENCHMARK
	float vertexShadingMs{};
	float TriangleAssemblingMs{};
	float RasterizationMs{};
#endif

	m_TotalVisibleNumTriangles = 0;
	for (MeshIdentifier& mi : m_MeshIdentifiers)
	{
		//Mesh Data
		const Mesh* pMesh = pMeshes[mi.Idx];
		const FMatrix4& worldMat = pMesh->GetWorldMatrix();
		const FMatrix4 worldViewProjectionMatrix = Transpose(viewProjectionMatrix * worldMat);
		const FMatrix3 rotationMatrix = Transpose(pMesh->GetRotationMatrix());

		//Update const data
		UpdateWorldMatrixDataAsync(worldMat, worldViewProjectionMatrix, rotationMatrix);
		cudaDeviceSynchronize();

		//TODO: can async copy (parts of) mesh buffers H2D
		//TODO: async & streams + find out what order is best, for cudaDevCpy and Malloc

#ifdef BENCHMARK
		StartTimer();
#endif

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

CPU_CALLABLE
void CUDARenderer::StartTimer()
{
	m_BenchMarker.StartTimer();
}

CPU_CALLABLE
float CUDARenderer::StopTimer()
{
	return m_BenchMarker.StopTimer();
}

#pragma endregion

#pragma region PRIVATE FUNCTIONS

CPU_CALLABLE
void CUDARenderer::InitCUDADeviceBuffers()
{
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
	//size = sizeof(unsigned int);
	//CheckErrorCuda(cudaMallocHost((void**)&m_h_pFrameBuffer, width * height * size));
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
	//Option 3: interpret depth buffer invertedly, so a depthvalue of 1.f is closest, and 0.f is furthest away from camera
	//>Option 4<: initialize and reset depthbuffer through additional kernel call, however this would be a lot of global memory accesses
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
void CUDARenderer::CopyMeshBuffers(const float* vertexBuffer, unsigned int numVertices, short stride, const unsigned int* indexBuffer, unsigned int numIndices, size_t meshIdx)
{
	//Copy Input Vertex Buffer
	CheckErrorCuda(cudaMemcpy(dev_IVertexBuffer[meshIdx], vertexBuffer, numVertices * stride, cudaMemcpyHostToDevice));
	//Copy Index Buffer
	CheckErrorCuda(cudaMemcpy(dev_IndexBuffer[meshIdx], indexBuffer, numIndices * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

CPU_CALLABLE
GPUTexturesCompact CUDARenderer::LoadMeshTextures(const std::string texturePaths[4], size_t meshIdx)
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
	return gpuTextures;
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
void CUDARenderer::UpdateCameraDataAsync(const FPoint3& camPos, const FVector3& camFwd)
{
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, camPos.data, sizeof(camPos)));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, camFwd.data, sizeof(camFwd), 3 * 4));
}

CPU_CALLABLE
void CUDARenderer::UpdateWorldMatrixDataAsync(const FMatrix4& worldMatrix, const FMatrix4& wvpMat, const FMatrix3& rotationMat)
{
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, worldMatrix.data, sizeof(worldMatrix), 6 * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, wvpMat.data, sizeof(wvpMat), 22 * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, rotationMat.data, sizeof(rotationMat), 38 * 4));
}

#pragma endregion

#pragma endregion

#pragma region KERNELS

//Kernel launch params:	numBlocks, numThreadsPerBlock, numSharedMemoryBytes, stream

#pragma region Clearing

GPU_KERNEL
void ClearDepthBufferKernel(int* dev_DepthBuffer, int value, const unsigned int width, const unsigned int height)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_DepthBuffer[pixelIdx] = value;
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
void ClearScreenKernel(PixelShade* dev_PixelShadeBuffer, const unsigned int width, const unsigned int height, unsigned int colour32)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_PixelShadeBuffer[pixelIdx].colour = colour32;
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
void ClearMutexBufferKernel(int* dev_MutexBuffer, const unsigned int width, const unsigned int height)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_MutexBuffer[pixelIdx] = 0;
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

	extern GPU_SHARED_MEMORY float sharedMemoryBuffer[];

	constexpr float* camPos = &dev_ConstMemory[0];
	constexpr float* worldMatrix = &dev_ConstMemory[6];
	constexpr float* WVPMatrix = &dev_ConstMemory[22];
	//TODO: rotationMatrix is just 3x3 of worldMatrix
	constexpr float* rotationMatrix = &dev_ConstMemory[38];

	//TODO: each thread should store 1 bank element at once for more coalesced access
	//instead of 1 thread storing 1 attribute from multiple banks to global memory

	const unsigned int vertexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vertexIdx < numVertices)
	{
		const IVertex& iVertex = dev_IVertices[vertexIdx];
		OVertex* pOVertex = &dev_OVertices[vertexIdx];
		float* pVecXYZ;

		// --- STEP 1 ---: Calculate Input Position to Ouput Position
		//for every 32 threads of vec3 (96 elements), a row of W elements is created (32 elements)

		//strided load into shared memory
		const unsigned int sharedMemVecIdx = threadIdx.x * 3 + (threadIdx.x / 32) * 32;
		pVecXYZ = &sharedMemoryBuffer[sharedMemVecIdx];
		const unsigned int sharedMemWIdx = threadIdx.x + 96 + (threadIdx.x / 32) * 96;
		float* pVecW = &sharedMemoryBuffer[sharedMemWIdx];

		memcpy(pVecXYZ, &iVertex.p, 12);
		//memory is now coalesced

		//calculate NDC (WVP * v.p.xyzw / w)
		CalculateOutputPosXYZW(WVPMatrix, pVecXYZ, pVecW); //calculate NDC (WVPMat)

		//divide xyz by w
		pVecXYZ[0] /= *pVecW;
		pVecXYZ[1] /= *pVecW;
		pVecXYZ[2] /= *pVecW;

		//store into global memory
		memcpy(&pOVertex->p, pVecXYZ, 12); //copy vec3 elements
		pOVertex->p.w = *pVecW; //copy w element

		// --- STEP 2 ---: Calculate ViewDirection

		memcpy(pVecXYZ, &iVertex.p, 12);

		CalculateOutputPosXYZ(worldMatrix, pVecXYZ); //calculate worldposition (worldMat)

		pVecXYZ[0] -= camPos[0];
		pVecXYZ[1] -= camPos[1];
		pVecXYZ[2] -= camPos[2];

		Normalize(reinterpret_cast<FVector3&>(*pVecXYZ)); //illegal memory access for some reason???

		memcpy(&pOVertex->vd, pVecXYZ, 12);
		__syncthreads(); //sync bc we don't use W value nomore

		// --- STEP 3 ---: Calculate Input Normal to Output Normal

		pVecXYZ = &sharedMemoryBuffer[threadIdx.x * 3];
		memcpy(pVecXYZ, &iVertex.n, 12);

		MultiplyMatVec(rotationMatrix, pVecXYZ, 3, 3); //calculate normal

		memcpy(&pOVertex->n, pVecXYZ, 12);

		// --- STEP 4 ---: Calculate Input Tangent to Output Tangent

		pVecXYZ = &sharedMemoryBuffer[threadIdx.x * 3];
		memcpy(pVecXYZ, &iVertex.tan, 12);

		MultiplyMatVec(rotationMatrix, pVecXYZ, 3, 3); //calculate tangent

		memcpy(&pOVertex->tan, pVecXYZ, 12);

		// --- STEP 5 ---: Copy UV and Colour

		//COLOUR
		pVecXYZ = &sharedMemoryBuffer[threadIdx.x * 3];
		memcpy(pVecXYZ, &iVertex.c, 12);
		__syncthreads();

		memcpy(&pOVertex->c, pVecXYZ, 12);

		//UV
		pVecXYZ = &sharedMemoryBuffer[threadIdx.x * 3]; //"padded" to avoid bank conflicts
		memcpy(pVecXYZ, &dev_IVertices[vertexIdx].uv, 8);
		__syncthreads();

		memcpy(&pOVertex->uv, pVecXYZ, 8);
	}
}

GPU_KERNEL
void TriangleAssemblerKernel(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer, unsigned int numIndices,
	OVertex* dev_OVertices, const PrimitiveTopology pt)
{
	//TODO: perform culling/clipping etc.
	//advantage of TriangleAssembly: each thread stores 1 triangle
	//many threads == many triangles processed at once

	//TODO: use shared memory to copy faster

	//'talk about naming gore, eh?
	const unsigned int indexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pt == PrimitiveTopology::TriangleList)
	{
		const unsigned int correctedIdx = (indexIdx * 3);
		if (correctedIdx < numIndices)
		{
			//Unnecessary for TriangleLists
			//triangle.isCulled = false;
			memcpy(&dev_Triangles[indexIdx], &dev_IndexBuffer[correctedIdx], sizeof(TriangleIdx));
		}
	}
	else //if (pt == PrimitiveTopology::TriangleStrip)
	{
		if (indexIdx < numIndices - 2)
		{
			//Necessary for TriangleStrips
			TriangleIdx triangle;
			const bool isOdd = (indexIdx % 2);
			//triangle.isCulled = false;

			memcpy(&triangle, &dev_IndexBuffer[indexIdx], sizeof(TriangleIdx));
			if (isOdd)
			{
				const unsigned int origIdx1 = triangle.idx1;
				triangle.idx1 = triangle.idx2;
				triangle.idx2 = origIdx1;
			}
			memcpy(&dev_Triangles[indexIdx], &triangle, sizeof(TriangleIdx));
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

	constexpr float* pCamFwd = &dev_ConstMemory[3];

	//extern GPU_SHARED_MEMORY float sharedMemoryBuffer[];

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

	//bool isDoubleSidedRendering = false;

	//is triangle visible according to cullingmode?
	//if (cm == CullingMode::BackFace)
	//{
	//	const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
	//	const float cullingValue = Dot(camFwd, faceNormal);
	//	if (cullingValue <= 0.f)
	//	{
	//		if (isDoubleSidedRendering)
	//		{
	//			OVertex origV1 = v1;
	//			v1 = v2;
	//			v2 = origV1;
	//		}
	//		else
	//		{
	//			return; //cull triangle
	//		}
	//	}
	//}
	//else if (cm == CullingMode::FrontFace)
	//{
	//	const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
	//	const float cullingValue = Dot(camFwd, faceNormal);
	//	if (cullingValue >= 0.f)
	//	{
	//		if (isDoubleSidedRendering)
	//		{
	//			OVertex origV1 = v1;
	//			v1 = v2;
	//			v2 = origV1;
	//		}
	//		else
	//		{
	//			return; //cull triangle
	//		}
	//	}
	//}
	//else if (cm == CullingMode::NoCulling)
	//{
	//}

	//if (!IsTriangleVisible(v0.p, v1.p, v2.p))
	//{
	//	return;
	//}

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
	//TODO: store PixelShade data column-based to avoid bank conflicts, but faster access?
	//GPU_SHARED_MEMORY PixelShade pixelShadeSharedMemory[width * height];

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
			const PixelShade& pixelShade = dev_PixelShadeBuffer[pixelIdx];
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
	//TODO: async in stream
	{
		constexpr int depthBufferResetValue = INT_MAX;
		const dim3 numThreadsPerBlock{ 32, 32 };
		const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
		ClearDepthBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_DepthBuffer, depthBufferResetValue, m_WindowHelper.Width, m_WindowHelper.Height);
	}

	{
		const size_t sizeInWords = m_WindowHelper.Width * m_WindowHelper.Height * (sizeof(PixelShade) / 4);
		constexpr unsigned int numThreadsPerBlock = 1024;
		const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
		ClearPixelShadeBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_PixelShadeBuffer, sizeInWords);
	}

	{
		////UNNECESSARY STEP: pixelshade stage will overwrite this anyway + more kernel launch overhead
		const RGBA rgba{ colour };
		const dim3 numThreadsPerBlock{ 32, 32 };
		const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
		//Needs to be called after ClearPixelShadeBufferKernel
		ClearScreenKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_PixelShadeBuffer, m_WindowHelper.Width, m_WindowHelper.Height, rgba.colour32);
		////Not necessary, since we overwrite the entire buffer every frame anyway
		//ClearFrameBufferKernel<<<numBlocks, numThreadsPerBlock>>>
		//	(dev_FrameBuffer, m_WindowHelper.Width, m_WindowHelper.Height, rgba.colour32);
	}

	{
		const dim3 numThreadsPerBlock{ 32, 32 };
		const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
		ClearMutexBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_MutexBuffer, m_WindowHelper.Width, m_WindowHelper.Height);
	}
}

CPU_CALLABLE
void CUDARenderer::VertexShader(const MeshIdentifier& mi)
{
	const unsigned int numVertices = mi.pMesh->GetVertexAmount();
	const IVertex* pIVertices = dev_IVertexBuffer[mi.Idx];
	OVertex* pOVertices = dev_OVertexBuffer[mi.Idx];

	//NOTE: NOT FOR COMPUTE CAPABILITY 6.1, stats may be higher
	//Max amount of shared memory per block: 49152 (48Kbs)
	//Max amount of threads per block/CTA: 2048
	//Max amount of blocks (dim.x): 2^31 - 1
	//Max amount of blocks (dim.yz): 65535

	constexpr unsigned int maxSharedMemoryPerBlock = 49152;
	constexpr unsigned int sharedMemoryNeeded = 128;
	unsigned int numThreadsPerBlock = maxSharedMemoryPerBlock / sharedMemoryNeeded;
	if (numVertices < numThreadsPerBlock)
	{
		numThreadsPerBlock = numVertices;
	}
	const unsigned int numBlocks = (numVertices + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	const unsigned int numSharedMemory = numThreadsPerBlock * sharedMemoryNeeded;
	VertexShaderKernel<<<numBlocks, numThreadsPerBlock, numSharedMemory>>>(
		pIVertices, pOVertices, numVertices);
}

CPU_CALLABLE
void CUDARenderer::TriangleAssembler(MeshIdentifier& mi)
{
	const unsigned int numTriangles = mi.TotalNumTriangles;
	const unsigned int numIndices = mi.pMesh->GetIndexAmount();
	const PrimitiveTopology topology = mi.pMesh->GetTopology();

	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (numTriangles + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	TriangleAssemblerKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_Triangles[mi.Idx], dev_IndexBuffer[mi.Idx], numIndices, 
		dev_OVertexBuffer[mi.Idx], topology);
}

CPU_CALLABLE
void CUDARenderer::Rasterizer(const MeshIdentifier& mi, const FVector3& camFwd, const CullingMode cm)
{
	const unsigned int numTriangles = mi.TotalNumTriangles;

	const dim3 numThreadsPerBlock = { 16, 16 };
	const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	RasterizerKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], numTriangles,
		dev_PixelShadeBuffer, dev_DepthBuffer, dev_MutexBuffer, mi.Textures,
		camFwd, cm, m_WindowHelper.Width, m_WindowHelper.Height);

	//RasterizerKernelCTA<<<numBlocks, numThreadsPerBlock>>>(
	//	dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], mi.TotalNumTriangles,
	//	dev_PixelShadeBuffer, dev_DepthBuffer, mi.Textures,
	//	m_WindowHelper.Width, m_WindowHelper.Height);
}

CPU_CALLABLE
void CUDARenderer::PixelShader(SampleState sampleState, bool isDepthColour)
{
	const dim3 numThreadsPerBlock{ 16, 16 };
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