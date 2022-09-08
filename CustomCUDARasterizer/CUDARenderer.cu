#include "PCH.h"
#include "CUDARenderer.h"
#include <vector>

//Project CUDA includes
#include "CUDATextureSampler.cuh"
#include "CUDAMatrixMath.cuh"
#include "RasterizerOperations.cu"
#include "CUDAStructs.h"
#include "CUDATextureManager.h"

#pragma region GLOBAL VARIABLES

//CONST DEVICE MEMORY - Does NOT have to be allocated or freed
constexpr unsigned int ConstMemorySize = 256;
constexpr unsigned int CamPosIdx = 0;
constexpr unsigned int CamFwdIdx = 3;
constexpr unsigned int WorldMatIdx = 6;
constexpr unsigned int WVPMatIdx = 22;
constexpr unsigned int RotMatIdx = 38;
constexpr unsigned int IsFinishedBinningIdx = 39;
GPU_CONST_MEMORY float dev_ConstMemory[ConstMemorySize];
//GPU_CONST_MEMORY float dev_CameraPos_const[sizeof(FPoint3) / sizeof(float)];
//GPU_CONST_MEMORY float dev_WVPMatrix_const[sizeof(FMatrix4) / sizeof(float)];
//GPU_CONST_MEMORY float dev_WorldMatrix_const[sizeof(FMatrix4) / sizeof(float)];
//GPU_CONST_MEMORY float dev_RotationMatrix_const[sizeof(FMatrix3) / sizeof(float)];

//DEVICE MEMORY - Does have to be allocated and freed
static unsigned int* dev_FrameBuffer{};
static int* dev_DepthBuffer{}; //defined as INTEGER type for atomicCAS to work properly
static int* dev_DepthMutexBuffer{};
static PixelShade* dev_PixelShadeBuffer{}; //(== fragmentbuffer)
static unsigned int* dev_NumVisibleTriangles{};
static unsigned int* dev_BinQueueSizes{};
static unsigned int* dev_BinQueues{};
static int* dev_BinQueueSizesMutexBuffer{}; //TODO: can be 16-bit to save memory?
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
CUDARenderer::CUDARenderer(const WindowHelper& windowHelper, IPoint2 numBins, IPoint2 binDim, unsigned int binQueueMaxSize)
	: m_WindowHelper{ windowHelper }
	, m_TotalNumTriangles{}
	, m_TotalVisibleNumTriangles{}
	//, m_h_pFrameBuffer{}
	, m_BenchMarker{}
	, m_BinDim{ binDim }
	, m_BinQueues{ (unsigned int)numBins.x, (unsigned int)numBins.y, binQueueMaxSize }
	, m_pCUDAMeshes{}
{
	AllocateCUDADeviceBuffers();
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

#ifdef BINNING
	std::cout << "\n---Triangle Binning enabled---\n";
	std::cout << "Number of bins: " << m_BinQueues.NumQueuesX << " , " << m_BinQueues.NumQueuesY << '\n';
	std::cout << "Dimension per bin: " << m_BinDim.x << " , " << m_BinDim.y << '\n';
	std::cout << "Queue size per bin: " << m_BinQueues.QueueMaxSize << '\n';
#else
	std::cout << "\n---Triangle Binning disabled---\n";
	std::cout << "Number of bins: 0, 0\n";
	std::cout << "Dimension per bin: 0, 0\n";
	std::cout << "Queue size per bin: 0, 0\n";
#endif
	std::cout << '\n';
}

CPU_CALLABLE
void CUDARenderer::LoadScene(const SceneGraph* pSceneGraph, const CUDATextureManager& tm)
{
	if (!pSceneGraph)
	{
		std::cout << "!CUDARenderer::LoadScene > Invalid scenegraph!\n";
		return;
	}
	m_TotalNumTriangles = 0;
	FreeAllCUDAMeshBuffers();
	const std::vector<Mesh*>& pMeshes = pSceneGraph->GetMeshes();
	for (const Mesh* pMesh : pMeshes)
	{
		CUDAMesh* pCudaMesh = AllocateCUDAMeshBuffers(pMesh);
		UpdateCUDAMeshTextures(pCudaMesh, tm);
		m_pCUDAMeshes.push_back(pCudaMesh);
		m_TotalNumTriangles += pCudaMesh->GetTotalNumTriangles();
	}
}

CPU_CALLABLE
void CUDARenderer::Render(const SceneManager& sm, const CUDATextureManager& tm, const Camera* pCamera)
{
	//Render Data
	const bool isDepthColour = sm.IsDepthColour();
	const SampleState sampleState = sm.GetSampleState();
	const CullingMode cm = sm.GetCullingMode();

	//Camera Data
	const FPoint3& camPos = pCamera->GetPosition();
	const FVector3& camFwd = pCamera->GetForward();
	const FMatrix4 viewMatrix = pCamera->GetViewMatrix();
	const FMatrix4& projectionMatrix = pCamera->GetProjectionMatrix();
	const FMatrix4 viewProjectionMatrix = projectionMatrix * viewMatrix;

	UpdateCameraDataAsync(camPos, camFwd);

#ifdef BENCHMARK
	float VertexShadingMs{};
	float TriangleAssemblingMs{};
#ifdef BINNING
	float TriangleBinningMs{};
#endif
	float RasterizationMs{};
	float PixelShadingMs{};
#endif

	m_TotalVisibleNumTriangles = 0;
	for (CUDAMesh* pCudaMesh : m_pCUDAMeshes)
	{
		//Mesh Data
		const Mesh* pMesh = pCudaMesh->GetMesh();
		const FMatrix4& worldMat = pMesh->GetWorldConst();
		const FMatrix4 worldViewProjectionMatrix = viewProjectionMatrix * worldMat;
		const FMatrix3& rotationMatrix = pMesh->GetRotationMatrix();

		//Update const data
		UpdateWorldMatrixDataAsync(worldMat, worldViewProjectionMatrix, rotationMatrix);
		CheckErrorCuda(cudaDeviceSynchronize());

		//TODO: can async copy (parts of) mesh buffers H2D
		//TODO: async & streams + find out what order is best, for cudaDevCpy and Malloc

#ifdef BENCHMARK
		StartTimer();
#endif

		//---STAGE 1---:  Perform Output Vertex Assembling
		VertexShader(pCudaMesh);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 1---

#ifdef BENCHMARK
		VertexShadingMs += StopTimer();
		StartTimer();
#endif

		//Reset number of visible triangles
		CheckErrorCuda(cudaMemset(dev_NumVisibleTriangles, 0, sizeof(unsigned int)));
		//int binnerStatus = 0;
		//CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, &binnerStatus, sizeof(int), IsFinishedBinningIdx * 4));

		//---STAGE 2---:  Perform Triangle Assembling
		TriangleAssembler(pCudaMesh, camFwd, cm);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 2---

		unsigned int& visibleNumTriangles = pCudaMesh->GetVisibleNumTriangles();
		CheckErrorCuda(cudaMemcpy(&visibleNumTriangles, dev_NumVisibleTriangles, 4, cudaMemcpyDeviceToHost));
		m_TotalVisibleNumTriangles += visibleNumTriangles;

#ifdef BENCHMARK
		TriangleAssemblingMs += StopTimer();
#endif

		//TODO: too many kernel launches
		//persistent kernel approach + global atomic flag value set in host
		const unsigned int numLoops = (visibleNumTriangles + (m_BinQueues.QueueMaxSize - 1)) / m_BinQueues.QueueMaxSize;
		for (unsigned int i{}; i < numLoops; ++i)
		{

#ifdef BINNING

#ifdef BENCHMARK
			StartTimer();
#endif

			//---STAGE 3---:  Perform Output Vertex Assembling
			TriangleBinner(pCudaMesh, i * m_BinQueues.QueueMaxSize);
			CheckErrorCuda(cudaDeviceSynchronize());
			//---END STAGE 3---

#ifdef BENCHMARK
			TriangleBinningMs += StopTimer();
			StartTimer();
#endif
#endif

			//TODO: Rasterization happens on a per-mesh basis instead of per-scenegraph?

			//---STAGE 4---: Peform Triangle Rasterization & interpolated fragment buffering
			Rasterizer(pCudaMesh, camFwd, cm);
			CheckErrorCuda(cudaDeviceSynchronize());

			//reset queue sizes
			const unsigned int size = m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY;
			CheckErrorCuda(cudaMemset(dev_BinQueueSizes, 0, size * 4));
			//reset queues
			CheckErrorCuda(cudaMemset(dev_BinQueues, 0, m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY * m_BinQueues.QueueMaxSize * 4));

			////TODO: measure time from binnerstream with event
			//WaitForStream(binnerStream);
			////IF BINNER STAGE HAS FINISHED, SET DIRTY FLAG
			//binnerStatus = 1;
			//CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, &binnerStatus, sizeof(int), IsFinishedBinningIdx * 4));
			//CheckErrorCuda(cudaDeviceSynchronize());

			//---END STAGE 4---

#ifdef BENCHMARK
		RasterizationMs += StopTimer();
#endif
		}
	}


#ifdef BENCHMARK
	StartTimer();
#endif

	//---STAGE 5---: Peform Pixel Shading
	PixelShader(sampleState, isDepthColour);
	CheckErrorCuda(cudaDeviceSynchronize());

	//---END STAGE 5---
#ifdef BENCHMARK
	PixelShadingMs = StopTimer();
	std::cout << "VS: " << VertexShadingMs 
		<< "ms | TA: " << TriangleAssemblingMs 
#ifdef BINNING
		<< "ms | Bin: " << TriangleBinningMs 
#endif
		<< "ms | Raster: " << RasterizationMs 
		<< "ms | PS: " << PixelShadingMs << "ms\r";
#endif
}

CPU_CALLABLE
void CUDARenderer::RenderAuto(const SceneManager& sm, const CUDATextureManager& tm, const Camera* pCamera)
{
#ifdef _DEBUG
	if (EnterValidRenderingState())
		exit(1);
#else
	EnterValidRenderingState();
#endif

	Render(sm, tm, pCamera);

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
	const size_t size = m_WindowHelper.Resolution.Width * m_WindowHelper.Resolution.Height * sizeof(unsigned int);
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
void CUDARenderer::AllocateCUDADeviceBuffers()
{
	size_t size{};
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

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
	cudaFree(dev_DepthMutexBuffer);
	cudaMalloc((void**)&dev_DepthMutexBuffer, width * height * size);
	cudaMemset(dev_DepthMutexBuffer, 0, width * height * size);

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

	//BINNING
	size = sizeof(unsigned int);
	cudaFree(dev_NumVisibleTriangles);
	cudaMalloc((void**)&dev_NumVisibleTriangles, size);
	cudaMemset(dev_NumVisibleTriangles, 0, size);

	//actual arrays of triangle indexes
	size = sizeof(unsigned int);
	cudaFree(dev_BinQueues);
	cudaMalloc((void**)&dev_BinQueues, m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY * m_BinQueues.QueueMaxSize * size);
	cudaMemset(dev_BinQueues, 0, m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY * m_BinQueues.QueueMaxSize * size);

	//actual size of arrays
	size = sizeof(unsigned int);
	cudaFree(dev_BinQueueSizes);
	cudaMalloc((void**)&dev_BinQueueSizes, m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY * size);
	cudaMemset(dev_BinQueueSizes, 0, m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY * size);

	//mutex buffer to alter bin queue sizes
	size = sizeof(int);
	cudaFree(dev_BinQueueSizesMutexBuffer);
	cudaMalloc((void**)&dev_BinQueueSizesMutexBuffer, m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY * size);
	cudaMemset(dev_BinQueueSizesMutexBuffer, 0, m_BinQueues.NumQueuesX * m_BinQueues.NumQueuesY * size);
}

CPU_CALLABLE
CUDAMesh* CUDARenderer::AllocateCUDAMeshBuffers(const Mesh* pMesh)
{
	const unsigned int meshIdx = (unsigned int)m_pCUDAMeshes.size();
	CUDAMesh* pCudaMesh = new CUDAMesh{ meshIdx, pMesh };
	return pCudaMesh;
}

CPU_CALLABLE
void CUDARenderer::FreeAllCUDAMeshBuffers()
{
	for (CUDAMesh* pCudaMesh : m_pCUDAMeshes)
	{
		FreeCUDAMeshBuffers(pCudaMesh);
	}
	m_pCUDAMeshes.clear();
}

CPU_CALLABLE
void CUDARenderer::FreeCUDAMeshBuffers(CUDAMesh* pCudaMesh)
{
	const auto it = std::find(m_pCUDAMeshes.cbegin(), m_pCUDAMeshes.cend(), pCudaMesh);
	if (it == m_pCUDAMeshes.cend())
		return;

	m_pCUDAMeshes.erase(it);
	delete pCudaMesh;
}

CPU_CALLABLE
void CUDARenderer::FreeCUDADeviceBuffers()
{
	//Free buffers
	CheckErrorCuda(cudaFree(dev_BinQueueSizesMutexBuffer));
	dev_BinQueueSizesMutexBuffer = nullptr;

	CheckErrorCuda(cudaFree(dev_BinQueueSizes));
	dev_BinQueueSizes = nullptr;

	CheckErrorCuda(cudaFree(dev_BinQueues));
	dev_BinQueues = nullptr;

	CheckErrorCuda(cudaFree(dev_NumVisibleTriangles));
	dev_NumVisibleTriangles = nullptr;

	CheckErrorCuda(cudaFree(dev_DepthMutexBuffer));
	dev_DepthMutexBuffer = nullptr;

	CheckErrorCuda(cudaFree(dev_DepthBuffer));
	dev_DepthBuffer = nullptr;

	CheckErrorCuda(cudaFree(dev_FrameBuffer));
	dev_FrameBuffer = nullptr;

	CheckErrorCuda(cudaFree(dev_PixelShadeBuffer));
	dev_PixelShadeBuffer = nullptr;

	//not allocated, but extra safety
	//CheckErrorCuda(cudaFreeHost(m_h_pFrameBuffer));
	//m_h_pFrameBuffer = nullptr;

	FreeAllCUDAMeshBuffers();
}

CPU_CALLABLE
void CUDARenderer::UpdateCameraDataAsync(const FPoint3& camPos, const FVector3& camFwd)
{
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, camPos.data, sizeof(camPos), CamPosIdx * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, camFwd.data, sizeof(camFwd), CamFwdIdx * 4));
}

CPU_CALLABLE
void CUDARenderer::UpdateWorldMatrixDataAsync(const FMatrix4& worldMatrix, const FMatrix4& wvpMat, const FMatrix3& rotationMat)
{
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, worldMatrix.data, sizeof(worldMatrix), WorldMatIdx * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, wvpMat.data, sizeof(wvpMat), WVPMatIdx * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, rotationMat.data, sizeof(rotationMat), RotMatIdx * 4));
}

CPU_CALLABLE
void CUDARenderer::UpdateCUDAMeshTextures(CUDAMesh* pCudaMesh, const CUDATextureManager& tm)
{
	const Mesh* pMesh = pCudaMesh->GetMesh();
	const int* texIds = pMesh->GetTextureIds();
	CUDATexturesCompact& textures = pCudaMesh->GetTextures();
	textures = GetCUDAMeshTextures(texIds, tm);
}

CPU_CALLABLE
CUDATexturesCompact CUDARenderer::GetCUDAMeshTextures(const int* texIds, const CUDATextureManager& tm)
{
	//Preload textures and fetch instead of creating a TexturesCompact object every frame in Render()
	//Instead of textures being fetched, alter them in CUDAMesh object
	//The actual TexuresCompact object gets copied to the kernel anyway (POD GPU data and ptrs)
	const CUDATexture* pDiff = tm.GetCUDATexture(texIds[Mesh::TextureID::Diffuse]);
	const CUDATexture* pNorm = tm.GetCUDATexture(texIds[Mesh::TextureID::Normal]);
	const CUDATexture* pSpec = tm.GetCUDATexture(texIds[Mesh::TextureID::Specular]);
	const CUDATexture* pGloss = tm.GetCUDATexture(texIds[Mesh::TextureID::Glossiness]);
	CUDATexturesCompact textures{};
	if (pDiff)
	{
		textures.Diff = *pDiff;
		if (pNorm)
		{
			textures.Norm = *pNorm;
			if (pSpec)
				textures.Spec = *pSpec;
			if (pGloss)
				textures.Gloss = *pGloss;
		}
	}
	return textures;
}

CPU_CALLABLE
void CUDARenderer::WaitForStream(cudaStream_t stream)
{
	cudaStreamSynchronize(stream);
}

CPU_CALLABLE
bool CUDARenderer::IsStreamFinished(cudaStream_t stream)
{
	const cudaError_t query = cudaStreamQuery(stream);
	return query == cudaSuccess;
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
void ClearDepthMutexBufferKernel(int* dev_MutexBuffer, const unsigned int width, const unsigned int height)
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
void VertexShaderKernelNaive(const IVertex* __restrict__ dev_IVertices, OVertex* dev_OVertices, unsigned int numVertices)
{
	//TODO: store matrix in top of shared memory for faster access
	//and offset shared memory access for threads
	//Potential problem: first warp might encounter bank conflicts?

	//The use of shared memory is not applicable here, (even though on-chip memory is faster)
	//- since the memory is not actually shared between threads
	//- most calculations are done within the thread's registers themselves (only 28 or 29 needed => < 32)
	//- the memory is not used multiple times

	//constexpr unsigned int paddedSizeOfIVertex = sizeof(IVertex) / 4 + 1;
	//extern GPU_SHARED_MEMORY float sharedMemoryBuffer[];

	const FPoint3& camPos = reinterpret_cast<const FPoint3&>(dev_ConstMemory[0]);
	const FMatrix4& worldMatrix = reinterpret_cast<const FMatrix4&>(dev_ConstMemory[6]);
	const FMatrix4& WVPMatrix = reinterpret_cast<const FMatrix4&>(dev_ConstMemory[22]);
	const FMatrix3& rotationMatrix = reinterpret_cast<const FMatrix3&>(dev_ConstMemory[38]);

	const unsigned int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexIdx < numVertices)
	{
		const IVertex& iVertex = dev_IVertices[vertexIdx];
		OVertex oVertex = GetNDCVertex(iVertex, WVPMatrix, worldMatrix, rotationMatrix, camPos);
		dev_OVertices[vertexIdx] = oVertex;
	}
}

GPU_KERNEL
void TriangleAssemblerKernel(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer, unsigned int numIndices,
	unsigned int* dev_NumVisibleTriangles,
	const OVertex* dev_OVertices, const PrimitiveTopology pt, const CullingMode cm, const FVector3 camFwd, 
	unsigned int width, unsigned int height)
{
	//advantage of TriangleAssembly: each thread stores 1 triangle
	//many threads == many triangles processed and/or culled at once

	//TODO: use shared memory to copy faster
	//data size of 9 shows no bank conflicts!
	//TriangleIdx can stay in local memory (registers)
	//DEPENDS ON REGISTER USAGE

	TriangleIdx triangle;

	//'talk about naming gore, eh?
	const unsigned int indexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pt == PrimitiveTopology::TriangleList)
	{
		const unsigned int correctedIdx = (indexIdx * 3);
		if (correctedIdx < numIndices)
		{
			memcpy(&triangle, &dev_IndexBuffer[correctedIdx], sizeof(TriangleIdx));
		}
		else
		{
			return;
		}
	}
	//TODO: get rid of these nasty if-statements
	else //if (pt == PrimitiveTopology::TriangleStrip)
	{
		if (indexIdx < numIndices - 2)
		{
			TriangleIdx triangle;
			memcpy(&triangle, &dev_IndexBuffer[indexIdx], sizeof(TriangleIdx));
			const bool isOdd = (indexIdx % 2);
			if (isOdd)
			{
				//swap without temp
				//TODO: what about indexes at UINT_MAX? => would just overflow and underflow back
				triangle.idx1 = triangle.idx1 + triangle.idx2;
				triangle.idx2 = triangle.idx1 - triangle.idx2;
				triangle.idx1 = triangle.idx1 - triangle.idx2;
			}
		}
		else
		{
			return;
		}
	}

	FPoint3 p0 = dev_OVertices[triangle.idx0].p.xyz;
	FPoint3 p1 = dev_OVertices[triangle.idx1].p.xyz;
	FPoint3 p2 = dev_OVertices[triangle.idx2].p.xyz;

	//PERFORM CULLING
	if (cm == CullingMode::BackFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ p1 - p0 }, FVector3{ p2 - p0 }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue <= 0.f)
		{
			return; //cull triangle
		}
	}
	else if (cm == CullingMode::FrontFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ p1 - p0 }, FVector3{ p2 - p0 }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue <= 0.f)
		{
			return; //cull triangle
		}
	}
	//else if (cm == CullingMode::NoCulling)
	//{
	//}

	//PERFORM CLIPPING
	if (!IsTriangleVisible(p0, p1, p2))
	{
		return;
	}
	
	//const float totalArea = abs(Cross(p0.xy - p1.xy, p2.xy - p0.xy));
	//if (totalArea <= 0.f)
	//{
	//	return; //cull away triangle
	//}

	const unsigned int triangleIdx = atomicAdd(dev_NumVisibleTriangles, 1); //returns old value
	memcpy(&dev_Triangles[triangleIdx], &triangle, sizeof(TriangleIdx));
}

GPU_KERNEL
void RasterizerPerTriangleKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices, 
	unsigned int numTriangles, PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, int* dev_DepthMutexBuffer, 
	CUDATexturesCompact textures, const unsigned int width, const unsigned int height)
{
	//constexpr unsigned int triangleSize = sizeof(OVertex) * 3 / 4;
	//extern GPU_SHARED_MEMORY float sharedMemoryBuffer[];

	//Each thread processes 1 triangle
	const unsigned int globalTriangleIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (!(globalTriangleIndex < numTriangles))
		return;

	const TriangleIdx triangleIdx = dev_Triangles[globalTriangleIndex];

	//Shared memory is laid out in a big row-list
	//const unsigned int triangleMemoryIdx = threadIdx.x * triangleSize;
	//OVertex& v0 = reinterpret_cast<OVertex&>(sharedMemoryBuffer[triangleMemoryIdx]);
	//OVertex& v1 = reinterpret_cast<OVertex&>(sharedMemoryBuffer[triangleMemoryIdx + (sizeof(OVertex) / 4)]);
	//OVertex& v2 = reinterpret_cast<OVertex&>(sharedMemoryBuffer[triangleMemoryIdx + (sizeof(OVertex) / 4) * 2]);

	//memcpy(&v0, &dev_OVertices[triangleIdx.idx0], sizeof(OVertex));
	//memcpy(&v1, &dev_OVertices[triangleIdx.idx1], sizeof(OVertex));
	//memcpy(&v2, &dev_OVertices[triangleIdx.idx2], sizeof(OVertex));

	OVertex v0 = dev_OVertices[triangleIdx.idx0];
	OVertex v1 = dev_OVertices[triangleIdx.idx1];
	OVertex v2 = dev_OVertices[triangleIdx.idx2];

	NDCToScreenSpace(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
	const BoundingBox bb = GetBoundingBox(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
	//Rasterize Screenspace triangle
	RasterizeTriangle(bb, v0, v1, v2, dev_DepthMutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);
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
	//TODO: if-statement is not necessary
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		RGBA rgba;
		if (isDepthColour)
		{
			rgba.r8 = 0; //For SDL: R and B values are swapped
			rgba.g8 = 0;
			rgba.b8 = (unsigned char)(Remap(dev_PixelShadeBuffer[pixelIdx].zInterpolated, 0.985f, 1.f) * 255);
			rgba.a8 = 0;
			dev_FrameBuffer[pixelIdx] = rgba.colour32;
		}
		else
		{
			const PixelShade& pixelShade = dev_PixelShadeBuffer[pixelIdx];
			const CUDATexturesCompact& textures = pixelShade.textures;
			if (textures.Diff.dev_pTex != 0)
			{
				RGBColor colour = ShadePixel(pixelShade.textures, pixelShade.uv, pixelShade.n, pixelShade.tan, pixelShade.vd, sampleState);
				rgba = colour; //== GetRGBAFromColour()
				dev_FrameBuffer[pixelIdx] = rgba.colour32;

				//normal visualisation
				//colour.r = pixelShade.n.x;
				//colour.g = pixelShade.n.y;
				//colour.b = pixelShade.n.z;
			}
			else
			{
				dev_FrameBuffer[pixelIdx] = pixelShade.colour;
			}
		}
	}
}

#pragma region Binning

GPU_KERNEL
void TriangleAssemblerAndBinnerKernel(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer, unsigned int numIndices,
	unsigned int* dev_NumVisibleTriangles, unsigned int* dev_BinQueueSizes, unsigned int* dev_BinQueues, int* dev_BinQueueSizesMutexBuffer,
	const OVertex* dev_OVertices, const PrimitiveTopology pt, const CullingMode cm, const FVector3 camFwd,
	IPoint2 numBins, IPoint2 binDim, unsigned int binQueueMaxSize, unsigned int width, unsigned int height)
{
	//advantage of TriangleAssembly: each thread stores 1 triangle
	//many threads == many triangles processed and/or culled at once

	//TODO: use shared memory to copy faster
	//data size of 9 shows no bank conflicts!
	//TriangleIdx can stay in local memory (registers)
	//DEPENDS ON REGISTER USAGE

	TriangleIdx triangle;

	//'talk about naming gore, eh?
	const unsigned int indexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pt == PrimitiveTopology::TriangleList)
	{
		const unsigned int correctedIdx = (indexIdx * 3);
		if (correctedIdx < numIndices)
		{
			memcpy(&triangle, &dev_IndexBuffer[correctedIdx], sizeof(TriangleIdx));
		}
		else
		{
			return;
		}
	}
	else //if (pt == PrimitiveTopology::TriangleStrip)
	{
		if (indexIdx < numIndices - 2)
		{
			TriangleIdx triangle;
			memcpy(&triangle, &dev_IndexBuffer[indexIdx], sizeof(TriangleIdx));
			const bool isOdd = (indexIdx % 2);
			if (isOdd)
			{
				//swap without temp
				//TODO: what about indexes at UINT_MAX? => would just overflow and back
				triangle.idx1 = triangle.idx1 + triangle.idx2;
				triangle.idx2 = triangle.idx1 - triangle.idx2;
				triangle.idx1 = triangle.idx1 - triangle.idx2;
			}
		}
		else
		{
			return;
		}
	}

	FPoint3 p0 = dev_OVertices[triangle.idx0].p.xyz;
	FPoint3 p1 = dev_OVertices[triangle.idx1].p.xyz;
	FPoint3 p2 = dev_OVertices[triangle.idx2].p.xyz;

	//PERFORM CULLING
	if (cm == CullingMode::BackFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ p1 - p0 }, FVector3{ p2 - p0 }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue <= 0.f)
		{
			return;
		}
	}
	else if (cm == CullingMode::FrontFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ p1 - p0 }, FVector3{ p2 - p0 }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue >= 0.f)
		{
			return; //cull triangle
		}
	}
	//else if (cm == CullingMode::NoCulling)
	//{
	//}

	//PERFORM CLIPPING
	if (!IsTriangleVisible(p0, p1, p2))
	{
		return;
	}

	const float totalArea = abs(Cross(p0.xy - p1.xy, p2.xy - p0.xy));
	if (totalArea <= 0.f)
	{
		return; //cull away triangle
	}

	const unsigned int triangleIdx = atomicAdd(dev_NumVisibleTriangles, 1); //returns old value
	memcpy(&dev_Triangles[triangleIdx], &triangle, sizeof(TriangleIdx));

	//BINNING
	//TODO: make separate kernel bc register usage is too high

	//assign to correct bin(s), with globalTriangleIdx corresponding to its bin
	//each bin is a part of the window (have multiple atomic buffers for each bin)

	NDCToScreenSpace(p0.xy, p1.xy, p2.xy, width, height);
	const BoundingBox triangleBb = GetBoundingBox(p0.xy, p1.xy, p2.xy, width, height);

	int binMinX = triangleBb.xMin / binDim.x; //most left bin
	int binMinY = triangleBb.yMin / binDim.y; //most bottom bin
	int binMaxX = triangleBb.xMax / binDim.x; //most right bin
	int binMaxY = triangleBb.yMax / binDim.y; //most top bin
	binMinX = ClampFast(binMinX, 0, numBins.x);
	binMinY = ClampFast(binMinY, 0, numBins.y);
	binMaxX = ClampFast(binMaxX, 0, numBins.x - 1);
	binMaxY = ClampFast(binMaxY, 0, numBins.y - 1);
	//This creates a grid of bins that overlap with triangle boundingbox

	//TODO: get all middle bin points in triangle polygon
	//TODO: get all intersecting rectangles of bins from all 3 triangle edges
	//https://stackoverflow.com/questions/16203760/how-to-check-if-line-segment-intersects-a-rectangle

	for (int y{ binMinY }; y <= binMaxY; ++y)
	{
		for (int x{ binMinX }; x <= binMaxX; ++x)
		{
			//atomically add triangle to bin queue
			const unsigned int binIdx = x + y * numBins.x;

			bool isDone = false;
			do
			{
				isDone = (atomicCAS(&dev_BinQueueSizesMutexBuffer[binIdx], 0, 1) == 0);
				if (isDone)
				{
					//critical section
					const unsigned int currQueueSize = dev_BinQueueSizes[binIdx];
					if (currQueueSize < binQueueMaxSize)
					{
						//insert triangle Idx in queue
						dev_BinQueues[binIdx * binQueueMaxSize + currQueueSize] = triangleIdx;
						//increase bin's queue size
						++dev_BinQueueSizes[binIdx];
					}
					dev_BinQueueSizesMutexBuffer[binIdx] = 0;
					//end of critical section
				}
			} while (!isDone);
		}
	}
}

GPU_KERNEL
void TriangleBinnerKernel(TriangleIdx* dev_Triangles, unsigned int numVisibleTriangles, unsigned int triangleIdxOffset,
	unsigned int* dev_BinQueueSizes, unsigned int* dev_BinQueues, int* dev_BinQueueSizesMutexBuffer,
	const OVertex* dev_OVertices, 
	IPoint2 numBins, IPoint2 binDim, unsigned int binQueueMaxSize, unsigned int width, unsigned int height)
{
	//TODO: use shared memory to copy faster
	//data size of 9 shows no bank conflicts!
	//TriangleIdx can stay in local memory (registers)
	//DEPENDS ON REGISTER USAGE

	//each thread bins 1 triangle
	const unsigned int triangleIdx = blockIdx.x * blockDim.x + threadIdx.x + triangleIdxOffset;
	if (triangleIdx < numVisibleTriangles)
	{
		const TriangleIdx triangle = dev_Triangles[triangleIdx];

		FPoint2 p0 = dev_OVertices[triangle.idx0].p.xy;
		FPoint2 p1 = dev_OVertices[triangle.idx1].p.xy;
		FPoint2 p2 = dev_OVertices[triangle.idx2].p.xy;

		//assign to correct bin(s), with globalTriangleIdx corresponding to its bin
		//each bin is a part of the window (have multiple atomic buffers for each bin)

		NDCToScreenSpace(p0, p1, p2, width, height);
		const BoundingBox triangleBb = GetBoundingBox(p0, p1, p2, width, height);

		int binMinX = (triangleBb.xMin) / binDim.x; //most left bin
		int binMinY = (triangleBb.yMin) / binDim.y; //most bottom bin
		int binMaxX = (triangleBb.xMax) / binDim.x; //most right bin
		int binMaxY = (triangleBb.yMax) / binDim.y; //most top bin
		binMinX = ClampFast(binMinX, 0, numBins.x);
		binMinY = ClampFast(binMinY, 0, numBins.y);
		binMaxX = ClampFast(binMaxX, 0, numBins.x - 1);
		binMaxY = ClampFast(binMaxY, 0, numBins.y - 1);
		//This creates a grid of bins that overlap with triangle boundingbox

		//TODO: get all middle bin points in triangle polygon
		//TODO: get all intersecting rectangles of bins from all 3 triangle edges
		//https://stackoverflow.com/questions/16203760/how-to-check-if-line-segment-intersects-a-rectangle

		for (int x{ binMinX }; x <= binMaxX; ++x)
		{
			for (int y{ binMinY }; y <= binMaxY; ++y)
			{
				//atomically add triangle to bin queue
				const unsigned int binIdx = x + y * numBins.x;

				bool isDone = false;
				do
				{
					isDone = (atomicCAS(&dev_BinQueueSizesMutexBuffer[binIdx], 0, 1) == 0);
					if (isDone)
					{
						//critical section
						const unsigned int currQueueSize = dev_BinQueueSizes[binIdx];

						if (currQueueSize < binQueueMaxSize)
						{
							//insert triangle Idx in queue
							dev_BinQueues[binIdx * binQueueMaxSize + currQueueSize] = triangleIdx;
							//increase bin's queue size
							++dev_BinQueueSizes[binIdx];

							//dev_BinQueueSizesMutexBuffer[binIdx] = 0; //release lock
						}
						dev_BinQueueSizesMutexBuffer[binIdx] = 0; //release lock
						//end of critical section
					}
				} while (!isDone);
			}
		}
	}
}

GPU_KERNEL
void RasterizerPerBinKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, int* dev_DepthMutexBuffer, CUDATexturesCompact textures,
	unsigned int* dev_BinQueues, unsigned int* dev_BinQueueSizes, int* dev_BinQueueSizesMutexBuffer,
	IPoint2 binDim, unsigned int binQueueMaxSize, const unsigned int width, const unsigned int height)
{
	//TODO: threads cooperatively store in shared memory
	//extern GPU_SHARED_MEMORY float sharedMemoryBuffer[];

	//PROGNOSIS: this will be slower and more threads will be "wasted" on the same potential triangle
	//BUT the main advantage is that big triangles will be eliminated and split up into smaller binned ones

	//each block processes 1 bin
	const unsigned int binIdx = blockIdx.x + blockIdx.y * gridDim.x;

	const unsigned int queueSize = dev_BinQueueSizes[binIdx];
	if (threadIdx.x < queueSize)
	{

		//each thread processes 1 triangle
		const unsigned int triangleIdx = dev_BinQueues[binIdx * binQueueMaxSize + threadIdx.x];

		const TriangleIdx triangle = dev_Triangles[triangleIdx];
		OVertex v0 = dev_OVertices[triangle.idx0];
		OVertex v1 = dev_OVertices[triangle.idx1];
		OVertex v2 = dev_OVertices[triangle.idx2];

		//3: rasterize triangle with binned bounding box (COARSE)

		NDCToScreenSpace(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
		const unsigned int minX = blockIdx.x * binDim.x;
		const unsigned int minY = blockIdx.y * binDim.y;
		const unsigned int maxX = minX + binDim.x;
		const unsigned int maxY = minY + binDim.y;
		const BoundingBox bb = GetBoundingBoxTiled(v0.p.xy, v1.p.xy, v2.p.xy, minX, minY, maxX, maxY);
		RasterizeTriangle(bb, v0, v1, v2, dev_DepthMutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);

		//BoundingBox bb;
		//bb.xMin = blockIdx.x * binDim.x;
		//bb.yMin = blockIdx.y * binDim.y;
		//bb.xMax = bb.xMin + binDim.x;
		//bb.yMax = bb.yMin + binDim.y;
		//RasterizeTriangle(bb, v0, v1, v2, dev_DepthMutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);

		//TODO: every thread in CTA processes a NxN tile of triangle in bin (fine rasterizer)
		//4: each thread in block does a 8x8 pixel area of triangle
		//each thread block does 1 triangle instead of each block does sizeofbinqueue triangles
	}
}

GPU_KERNEL
void RasterizerPerTileKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, int* dev_DepthMutexBuffer, CUDATexturesCompact textures,
	unsigned int* dev_BinQueues, unsigned int* dev_BinQueueSizes, int* dev_BinQueueSizesMutexBuffer, IPoint2 binDim, unsigned int binQueueMaxSize, 
	const unsigned int pixelCoverageX, const unsigned int pixelCoverageY, const unsigned int width, const unsigned int height)
{
	//TODO: blocksize.x == binQueueMaxSize

	//extern GPU_SHARED_MEMORY unsigned int triangleQueueBuffer[];
	//GPU_SHARED_MEMORY float triangleData[(sizeof(OVertex) / 4) * 3];

	//each block processes 1 bin
	const unsigned int binIdx = blockIdx.x + blockIdx.y * gridDim.x;
	const unsigned int queueSize = dev_BinQueueSizes[binIdx];
	//const unsigned int queueSize = binQueueMaxSize;

	//every thread covers a NxN pixel area
	const unsigned int minX = blockIdx.x * binDim.x;
	const unsigned int minY = blockIdx.y * binDim.y;
	const unsigned int maxX = minX + binDim.x;
	const unsigned int maxY = minY + binDim.y;

	BoundingBox bb;
	bb.xMin = minX + threadIdx.x * pixelCoverageX;
	bb.yMin = minY + threadIdx.y * pixelCoverageY;
	bb.xMax = bb.xMin + pixelCoverageX;
	bb.yMax = bb.yMin + pixelCoverageY;

	bb.xMin = ClampFast(bb.xMin, (short)minX, (short)maxX);
	bb.xMax = ClampFast(bb.xMax, (short)minX, (short)maxX);
	bb.yMin = ClampFast(bb.yMin, (short)minY, (short)maxY);
	bb.yMax = ClampFast(bb.yMax, (short)minY, (short)maxY);

	for (unsigned int currQueueIdx{}; currQueueIdx < queueSize; ++currQueueIdx)
	{
		const unsigned int triangleIdx = dev_BinQueues[binIdx * binQueueMaxSize + currQueueIdx];
		const TriangleIdx triangle = dev_Triangles[triangleIdx];
		OVertex v0 = dev_OVertices[triangle.idx0];
		OVertex v1 = dev_OVertices[triangle.idx1];
		OVertex v2 = dev_OVertices[triangle.idx2];

		NDCToScreenSpace(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
		//Rasterize Screenspace triangle
		RasterizeTriangle(bb, v0, v1, v2, dev_DepthMutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);

		////acts as a block barrier for shared memory usage
		//__syncthreads();
	}
}

#pragma endregion

#pragma region DEPRECATED

GPU_KERNEL
void VertexShaderKernelPrototype(const IVertex* __restrict__ dev_IVertices, OVertex* dev_OVertices, unsigned int numVertices)
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
	constexpr float* rotationMatrix = &dev_ConstMemory[38];

	//TODO: each thread should store 1 bank element at once for more coalesced access
	//instead of 1 thread storing 1 attribute from multiple banks to global memory

	//threadIdx.x: [0, 31]
	//threadIdx.y: [0, 7]
	const unsigned int vertexIdx = (blockIdx.x * (blockDim.x * blockDim.y)) + threadIdx.x + (threadIdx.y * blockDim.x);
	if (vertexIdx < numVertices)
	{
		const IVertex& iVertex = dev_IVertices[vertexIdx];
		OVertex* pOVertex = &dev_OVertices[vertexIdx];

		//TODO: store W component in local memory???
		//TODO: register usage is above 32, mainly due to matrixmath functions
		//also some used for shared memory and pointers

		// --- STEP 1 ---: Calculate Input Position to Ouput Position
		//for every 32 threads of vec3 (96 elements), a row of W elements is created (32 elements)

		//strided load into shared memory
		const unsigned int warpSharedMemIdx = threadIdx.y * 128;
		unsigned int sharedMemVecIdx = threadIdx.x * 3 + warpSharedMemIdx;
		float* pVecXYZ = &sharedMemoryBuffer[sharedMemVecIdx];
		const unsigned int sharedMemWIdx = threadIdx.x + 96 + warpSharedMemIdx;
		float* pVecW = &sharedMemoryBuffer[sharedMemWIdx];

		//memory is now coalesced
		memcpy(pVecXYZ, &iVertex.p, 12);
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
		Normalize(reinterpret_cast<FVector3&>(*pVecXYZ));

		memcpy(&pOVertex->vd, pVecXYZ, 12);
		__syncthreads(); //sync bc we don't use W value nomore

		//shared memory is now used 
		sharedMemVecIdx = threadIdx.x * 3 + threadIdx.y * 96;
		pVecXYZ = &sharedMemoryBuffer[sharedMemVecIdx];

		// --- STEP 3 ---: Calculate Input Normal to Output Normal

		memcpy(pVecXYZ, &iVertex.n, 12);
		MultiplyMatVec(rotationMatrix, pVecXYZ, 3, 3); //calculate normal
		memcpy(&pOVertex->n, pVecXYZ, 12);

		// --- STEP 4 ---: Calculate Input Tangent to Output Tangent

		memcpy(pVecXYZ, &iVertex.tan, 12);
		MultiplyMatVec(rotationMatrix, pVecXYZ, 3, 3); //calculate tangent
		memcpy(&pOVertex->tan, pVecXYZ, 12);

		// --- STEP 5 ---: Copy UV and Colour

		//COLOUR
		memcpy(pVecXYZ, &iVertex.c, 12);
		memcpy(&pOVertex->c, pVecXYZ, 12);

		//UV
		//pVecXYZ is "padded" UV to avoid bank conflicts
		memcpy(pVecXYZ, &iVertex.uv, 8);
		memcpy(&pOVertex->uv, pVecXYZ, 8);
	}
}

GPU_KERNEL
void TriangleAssemblerKernelOld(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer,
	unsigned int numIndices, const PrimitiveTopology pt)
{
	//TODO: perform culling/clipping etc.
	//advantage of TriangleAssembly: each thread stores 1 triangle
	//many threads == many triangles processed at once

	//TODO: global to shared to global?
	//TODO: local copy to global vs global to global?

	//'talk about naming gore, eh?
	const unsigned int indexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pt == PrimitiveTopology::TriangleList)
	{
		const unsigned int correctedIdx = (indexIdx * 3);
		if (correctedIdx < numIndices)
		{
			memcpy(&dev_Triangles[indexIdx], &dev_IndexBuffer[correctedIdx], sizeof(TriangleIdx));
		}
	}
	else //if (pt == PrimitiveTopology::TriangleStrip)
	{
		if (indexIdx < numIndices - 2)
		{
			TriangleIdx triangle;
			memcpy(&triangle, &dev_IndexBuffer[indexIdx], sizeof(TriangleIdx));
			const bool isOdd = (indexIdx % 2);
			if (isOdd)
			{
				//swap without temp
				//TODO: what about indexes at UINT_MAX? => would just overflow and back
				triangle.idx1 = triangle.idx1 + triangle.idx2;
				triangle.idx2 = triangle.idx1 - triangle.idx2;
				triangle.idx1 = triangle.idx1 - triangle.idx2;
			}
			memcpy(&dev_Triangles[indexIdx], &triangle, sizeof(TriangleIdx));
		}
	}
}

GPU_KERNEL
void RasterizerPerPixelKernel(const TriangleIdx* __restrict__ dev_Triangles, const OVertex* __restrict__ const dev_OVertices, unsigned int numTriangles,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, CUDATexturesCompact textures,
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

		NDCToScreenSpace(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
		const BoundingBox bb = GetBoundingBox(v0.p.xy, v1.p.xy, v2.p.xy, width, height);

		if (!IsPixelInBoundingBox(pixel, bb))
		{
			return;
		}

		//Rasterize pixel
		RasterizePixel(pixel, v0, v1, v2, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);
	}
}

GPU_KERNEL
void RasterizerPerTriangleKernelOld(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices, unsigned int numTriangles,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, int* dev_MutexBuffer, CUDATexturesCompact textures,
	const FVector3 camFwd, const CullingMode cm, const unsigned int width, const unsigned int height)
{
	//constexpr unsigned int triangleSize = sizeof(OVertex) * 3 / 4;
	extern GPU_SHARED_MEMORY float sharedMemoryBuffer[];

	//Every thread processes 1 single triangle for now
	const unsigned int globalTriangleIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (!(globalTriangleIndex < numTriangles))
		return;

	const TriangleIdx triangleIdx = dev_Triangles[globalTriangleIndex];

	//Shared memory is laid out in a big row-list
	//const unsigned int triangleMemoryIdx = threadIdx.x * triangleSize;
	//OVertex& v0 = reinterpret_cast<OVertex&>(sharedMemoryBuffer[triangleMemoryIdx]);
	//OVertex& v1 = reinterpret_cast<OVertex&>(sharedMemoryBuffer[triangleMemoryIdx + (sizeof(OVertex) / 4)]);
	//OVertex& v2 = reinterpret_cast<OVertex&>(sharedMemoryBuffer[triangleMemoryIdx + (sizeof(OVertex) / 4) * 2]);

	//memcpy(&v0, &dev_OVertices[triangleIdx.idx0], sizeof(OVertex));
	//memcpy(&v1, &dev_OVertices[triangleIdx.idx1], sizeof(OVertex));
	//memcpy(&v2, &dev_OVertices[triangleIdx.idx2], sizeof(OVertex));

	OVertex v0 = dev_OVertices[triangleIdx.idx0];
	OVertex v1 = dev_OVertices[triangleIdx.idx1];
	OVertex v2 = dev_OVertices[triangleIdx.idx2];

	//bool isDoubleSidedRendering = false;

	//is triangle visible according to cullingmode?
	if (cm == CullingMode::BackFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue <= 0.f)
		{
			//if (isDoubleSidedRendering)
			//{
			//	OVertex origV1 = v1;
			//	v1 = v2;
			//	v2 = origV1;
			//}
			//else
			//{
			return; //cull triangle
		//}
		}
	}
	else if (cm == CullingMode::FrontFace)
	{
		const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
		const float cullingValue = Dot(camFwd, faceNormal);
		if (cullingValue >= 0.f)
		{
			//if (isDoubleSidedRendering)
			//{
			//	OVertex origV1 = v1;
			//	v1 = v2;
			//	v2 = origV1;
			//}
			//else
			//{
			return; //cull triangle
		//}
		}
	}
	//else if (cm == CullingMode::NoCulling)
	//{
	//}

	if (!IsTriangleVisible(v0.p, v1.p, v2.p))
	{
		return;
	}

	NDCToScreenSpace(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
	const BoundingBox bb = GetBoundingBox(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
	//Rasterize Screenspace triangle
	RasterizeTriangle(bb, v0, v1, v2, dev_MutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);
}

#pragma endregion

#pragma region TESTING

GPU_KERNEL
void TextureTestKernel(unsigned int* dev_FrameBuffer, CUDATextureCompact texture, const unsigned int width, const unsigned int height)
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
void DrawTextureGlobalKernel(unsigned int* dev_FrameBuffer, CUDATextureCompact texture, bool isStretchedToWindow,
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
		RGBColor sample = CUDATextureSampler::Sample(texture, uv, sampleState);
		RGBA rgba = sample;
		dev_FrameBuffer[pixelIdx] = rgba.colour32;
	}
}

#pragma endregion

#pragma endregion

#pragma region KERNEL LAUNCHERS

CPU_CALLABLE
void CUDARenderer::Clear(const RGBColor& colour)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

	//TODO: async in stream
	{
		constexpr int depthBufferResetValue = INT_MAX;
		const dim3 numThreadsPerBlock{ 32, 32 };
		const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
		ClearDepthBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_DepthBuffer, depthBufferResetValue, width, height);
	}

	{
		const size_t sizeInWords = width * height * (sizeof(PixelShade) / 4);
		constexpr unsigned int numThreadsPerBlock = 512;
		const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
		ClearPixelShadeBufferKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_PixelShadeBuffer, sizeInWords);
	}

	{
		////UNNECESSARY STEP: pixelshade stage will overwrite this anyway + more kernel launch overhead
		const RGBA rgba{ colour };
		const dim3 numThreadsPerBlock{ 32, 32 };
		const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
		//Needs to be called after ClearPixelShadeBufferKernel
		ClearScreenKernel<<<numBlocks, numThreadsPerBlock>>>
			(dev_PixelShadeBuffer, width, height, rgba.colour32);
		////Not necessary, since we overwrite the entire buffer every frame anyway
		//ClearFrameBufferKernel<<<numBlocks, numThreadsPerBlock>>>
		//	(dev_FrameBuffer, width, height, rgba.colour32);
	}

	{
		////UNNECESSARY STEP: mutexbuffer should always revert to base initialized state, otherwise deadlocks would occur
		//const dim3 numThreadsPerBlock{ 32, 32 };
		//const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
		//ClearDepthMutexBufferKernel<<<numBlocks, numThreadsPerBlock>>>
		//	(dev_DepthMutexBuffer, width, height);
	}
}

CPU_CALLABLE
void CUDARenderer::VertexShader(const CUDAMesh* pCudaMesh)
{
	const Mesh* pMesh = pCudaMesh->GetMesh();
	const unsigned int numVertices = pMesh->GetNumVertices();

	const IVertex* dev_IVertexBuffer = pCudaMesh->GetDevIVertexBuffer();
	OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();

	//constexpr unsigned int paddedSizeOfIVertex = sizeof(OVertex) / 4 + 1;
	//constexpr unsigned int sharedMemoryNeededPerThread = paddedSizeOfIVertex * 4;
	//constexpr unsigned int sharedMemoryNeededPerWarp = sharedMemoryNeededPerThread * 32;
	constexpr unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (numVertices + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	//const unsigned int numSharedMemory = numThreadsPerBlock * sharedMemoryNeededPerThread;
	VertexShaderKernelNaive<<<numBlocks, numThreadsPerBlock>>>(
		dev_IVertexBuffer, dev_OVertexBuffer, numVertices);

	//NOTE: NOT FOR COMPUTE CAPABILITY 6.1, stats may be higher
	//Max amount of shared memory per block: 49152 (48Kbs)
	//Max amount of threads per block/CTA: 2048
	//Max amount of blocks (dim.x): 2^31 - 1
	//Max amount of blocks (dim.yz): 65535
	//Max amount of blocks per SM: 8

	////constexpr unsigned int maxSharedMemoryPerBlock = 49152;
	//constexpr unsigned int sharedMemoryNeededPerThread = 4 * sizeof(float);
	//constexpr unsigned int sharedMemoryNeededPerWarp = sharedMemoryNeededPerThread * 32;
	//const dim3 numThreadsPerBlock{ 32, 8 }; //256
	//const unsigned int numThreadsPerBlockTotal = numThreadsPerBlock.x * numThreadsPerBlock.y;
	//const unsigned int numBlocks = (numVertices + (numThreadsPerBlockTotal - 1)) / numThreadsPerBlockTotal;
	//const unsigned int numSharedMemory = numThreadsPerBlock.y * sharedMemoryNeededPerWarp;
	//VertexShaderKernelPrototype<<<numBlocks, numThreadsPerBlock, numSharedMemory>>>(
	//	dev_IVertexBuffer[mi.Idx], dev_OVertexBuffer[mi.Idx], numVertices);
}

CPU_CALLABLE
void CUDARenderer::TriangleAssembler(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm, cudaStream_t stream)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

	const unsigned int numTriangles = pCudaMesh->GetTotalNumTriangles();
	const Mesh* pMesh = pCudaMesh->GetMesh();
	const unsigned int numIndices = pMesh->GetNumIndices();
	const PrimitiveTopology topology = pMesh->GetTopology();

	const unsigned int* dev_IndexBuffer = pCudaMesh->GetDevIndexBuffer();
	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (numTriangles + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	TriangleAssemblerKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, dev_IndexBuffer, numIndices, dev_NumVisibleTriangles,
		dev_OVertexBuffer, topology, cm, camFwd,
		width, height);

	//TriangleAssemblerKernelOld<<<numBlocks, numThreadsPerBlock>>>(
	//	dev_Triangles[mi.Idx], dev_IndexBuffer[mi.Idx], 
	//	numIndices, topology);
}

CPU_CALLABLE
void CUDARenderer::TriangleBinner(const CUDAMesh* pCudaMesh, const unsigned int triangleIdxOffset, cudaStream_t stream)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;
	const unsigned int numVisibleTriangles = pCudaMesh->GetVisibleNumTrianglesConst();
	const IPoint2 numBins = { (int)m_BinQueues.NumQueuesX, (int)m_BinQueues.NumQueuesY };

	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

	const unsigned int numThreadsPerBlock = m_BinQueues.QueueMaxSize;
	const unsigned int numBlocks = 1;
	TriangleBinnerKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, numVisibleTriangles, triangleIdxOffset,
		dev_BinQueueSizes, dev_BinQueues, dev_BinQueueSizesMutexBuffer,
		dev_OVertexBuffer,
		numBins, m_BinDim, m_BinQueues.QueueMaxSize, width, height);
}

CPU_CALLABLE
void CUDARenderer::TriangleAssemblerAndBinner(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm, cudaStream_t stream)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;
	const Mesh* pMesh = pCudaMesh->GetMesh();
	const unsigned int numIndices = pMesh->GetNumIndices();
	const PrimitiveTopology topology = pMesh->GetTopology();
	const unsigned int numTriangles = pCudaMesh->GetTotalNumTriangles();
	const IPoint2 numBins = { (int)m_BinQueues.NumQueuesX, (int)m_BinQueues.NumQueuesY };

	const unsigned int* dev_IndexBuffer = pCudaMesh->GetDevIndexBuffer();
	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (numTriangles + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	TriangleAssemblerAndBinnerKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, dev_IndexBuffer, numIndices, dev_NumVisibleTriangles,
		dev_BinQueueSizes, dev_BinQueues, dev_BinQueueSizesMutexBuffer,
		dev_OVertexBuffer, topology, cm, camFwd,
		numBins, m_BinDim, m_BinQueues.QueueMaxSize, width, height);
}

CPU_CALLABLE
void CUDARenderer::Rasterizer(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm, cudaStream_t stream)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;
	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	const CUDATexturesCompact& textures = pCudaMesh->GetTexturesConst();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

#ifdef BINNING
#ifdef FINERASTER
	const dim3 numThreadsPerBlock = { 16, 16 };
	const dim3 numBlocks = { m_BinQueues.NumQueuesX, m_BinQueues.NumQueuesY };
	//const unsigned int numSharedMemory = m_BinQueues.QueueMaxSize * 4 + 4; //queue array + 1 queue size

	//pixel coverage per thread
	const unsigned int pixelCoverageX = m_BinDim.x / numThreadsPerBlock.x;
	const unsigned int pixelCoverageY = m_BinDim.y / numThreadsPerBlock.y;
	RasterizerPerTileKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, dev_OVertexBuffer,
		dev_PixelShadeBuffer, dev_DepthBuffer, dev_DepthMutexBuffer, textures,
		dev_BinQueues, dev_BinQueueSizes, dev_BinQueueSizesMutexBuffer, m_BinDim,
		m_BinQueues.QueueMaxSize, pixelCoverageX, pixelCoverageY, width, height);
	//TODO: each block iterates through entire queue array, how about each block only does 1 triangle of 1 queue *smirk*
#else
	constexpr unsigned int numThreadsPerBlock = 256;
	const dim3 numBlocks = { m_BinQueues.NumQueuesX, m_BinQueues.NumQueuesY };
	RasterizerPerBinKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx],
		dev_PixelShadeBuffer, dev_DepthBuffer, dev_DepthMutexBuffer, textures,
		dev_BinQueues, dev_BinQueueSizes, dev_BinQueueSizesMutexBuffer, m_BinDim,
		m_BinQueues.QueueMaxSize, width, height);
#endif
#else
	constexpr unsigned int numThreadsPerBlock = 256;
	const unsigned int numTriangles = mi.VisibleNumTriangles;
	const unsigned int numBlocks = (numTriangles + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	RasterizerPerTriangleKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], numTriangles,
		dev_PixelShadeBuffer, dev_DepthBuffer, dev_DepthMutexBuffer, textures,
		width, height);
#endif

	//SHARED MEMORY ATTEMPT
	//shared memory needed per triangle = 3 * OVertex (216 bytes)
	//shared memory per block: 49152 bytes
	//Thus 227.555 triangles per 48Kbs
	//32 * 7 = 224 (only 3.555 difference)
	//224 * 216 = 48384 bytes (768 bytes 'wasted')

	//constexpr unsigned int sizeOfTriangle = sizeof(OVertex) * 3;
	//constexpr unsigned int maxSharedMemoryPerBlock = 49152;
	//constexpr unsigned int numThreadsPerBlock = (maxSharedMemoryPerBlock / sizeOfTriangle) - (maxSharedMemoryPerBlock / sizeOfTriangle) % 32;
	//constexpr unsigned int numThreadsPerBlock = 128;
	//constexpr unsigned int numSharedMemory = numThreadsPerBlock * sizeOfTriangle;
	//const unsigned int numBlocks = (numTriangles + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	//RasterizerPerTriangleKernel<<<numBlocks, numThreadsPerBlock, numSharedMemory>>>(
	//	dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], numTriangles,
	//	dev_PixelShadeBuffer, dev_DepthBuffer, dev_MutexBuffer, textures,
	//	width, height);

	//RasterizerPerTriangleKernelOld<<<numBlocks, numThreadsPerBlock, numSharedMemory>>>(
	//	dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], numTriangles,
	//	dev_PixelShadeBuffer, dev_DepthBuffer, dev_MutexBuffer, textures,
	//	camFwd, cm, width, height);

	//const dim3 numThreadsPerBlock = { 16, 16 };
	//const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	//RasterizerPerPixelKernel<<<numBlocks, numThreadsPerBlock>>>(
	//	dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], numTriangles,
	//	dev_PixelShadeBuffer, dev_DepthBuffer, textures,
	//	width, height);
}

CPU_CALLABLE
void CUDARenderer::PixelShader(SampleState sampleState, bool isDepthColour)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
	PixelShaderKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_FrameBuffer, dev_PixelShadeBuffer, sampleState, isDepthColour,
		width, height);
}

CPU_CALLABLE
void CUDARenderer::DrawTexture(char* tP)
{
	CUDATexture texture{};
	texture.Create(tP);

	EnterValidRenderingState();

	const CUDATextureCompact texCompact{ texture };

	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;
	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
	TextureTestKernel<<<numBlocks, numThreadsPerBlock>>>(dev_FrameBuffer, texCompact, width, height);

	Present();
}

CPU_CALLABLE
void CUDARenderer::DrawTextureGlobal(char* tp, bool isStretchedToWindow, SampleState sampleState)
{
	CUDATexture texture{};
	texture.Create(tp);

	EnterValidRenderingState();

	const CUDATextureCompact texCompact{ texture };

	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;
	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
	DrawTextureGlobalKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_FrameBuffer, texCompact, isStretchedToWindow,
		sampleState, width, height);

	Present();
}

CPU_CALLABLE
void CUDARenderer::WarmUp()
{
	ClearDepthBufferKernel<<<0, 0>>>(nullptr, 0, 0, 0);
	ClearFrameBufferKernel<<<0, 0>>>(nullptr, 0, 0, 0);
	ClearPixelShadeBufferKernel<<<0, 0>>>(nullptr, 0);
	ClearDepthMutexBufferKernel <<<0, 0>>>(nullptr, 0, 0);
	VertexShaderKernelNaive<<<0, 0>>>(nullptr, nullptr, 0);
	TriangleAssemblerKernelOld<<<0, 0>>>(nullptr, nullptr, 0, (PrimitiveTopology)0);
	RasterizerPerTriangleKernelOld<<<0, 0>>>(nullptr, nullptr, 0, nullptr, nullptr, nullptr, {}, {}, (CullingMode)0, 0, 0);
	PixelShaderKernel<<<0, 0>>>(nullptr, nullptr, SampleState(0), false, 0, 0);
}

#pragma endregion