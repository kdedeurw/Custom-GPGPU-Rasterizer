#include "PCH.h"
#include "CUDARenderer.h"
#include <vector>
#include <SDL.h>

//Project CUDA includes
#include "CUDATextureSampler.cuh"
#include "CUDAMatrixMath.cuh"
#include "CUDAROPs.cu"

#pragma region GLOBAL VARIABLES

constexpr unsigned int ConstMemorySize = 256;
constexpr unsigned int CamPosIdx = 0;
constexpr unsigned int CamFwdIdx = 3;
constexpr unsigned int WorldMatIdx = 6;
constexpr unsigned int WVPMatIdx = 22;
constexpr unsigned int RotMatIdx = 38;
constexpr unsigned int IsFinishedBinningIdx = 39;

//CONST DEVICE MEMORY - Does NOT have to be allocated or freed
GPU_CONST_MEMORY float dev_ConstMemory[ConstMemorySize];
//GPU_CONST_MEMORY float dev_CameraPos_const[sizeof(FPoint3) / sizeof(float)];
//GPU_CONST_MEMORY float dev_WVPMatrix_const[sizeof(FMatrix4) / sizeof(float)];
//GPU_CONST_MEMORY float dev_WorldMatrix_const[sizeof(FMatrix4) / sizeof(float)];
//GPU_CONST_MEMORY float dev_RotationMatrix_const[sizeof(FMatrix3) / sizeof(float)];

#pragma endregion

//--------------------------

CPU_CALLABLE
CUDARenderer::CUDARenderer(const WindowHelper& windowHelper)
	: m_TotalVisibleNumTriangles{}
	, m_WindowHelper{ windowHelper }
	, m_Dev_NumVisibleTriangles{}
	, m_BinDim{}
	, m_BenchMarker{}
	, m_BinQueues{}
	, m_CUDAWindowHelper{}
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
	std::cout << "Number of bins: " << m_BinQueues.GetNumQueuesX() << " , " << m_BinQueues.GetNumQueuesY() << '\n';
	std::cout << "Dimension per bin: " << m_BinDim.x << " , " << m_BinDim.y << '\n';
	std::cout << "Queue size per bin: " << m_BinQueues.GetQueueMaxSize() << '\n';
#else
	std::cout << "\n---Triangle Binning disabled---\n";
	std::cout << "Number of bins: 0, 0\n";
	std::cout << "Dimension per bin: 0, 0\n";
	std::cout << "Queue size per bin: 0, 0\n";
#endif
	std::cout << '\n';
}

CPU_CALLABLE
void CUDARenderer::SetupBins(const IPoint2& numBins, const IPoint2& binDim, unsigned int binQueueMaxSize)
{
	m_BinQueues.Init(numBins.x, numBins.y, binQueueMaxSize);
	m_BinDim = binDim;
}

CPU_CALLABLE
void CUDARenderer::Render(const CUDASceneGraph& scene, const CUDATextureManager& tm, const Camera& camera)
{
	//Render Data
	const VisualisationState visualisationState = scene.GetVisualisationState();
	const SampleState sampleState = scene.GetSampleState();
	const CullingMode cm = scene.GetCullingMode();

	//Camera Data
	const FPoint3& camPos = camera.GetPosition();
	const FVector3& camFwd = camera.GetForward();
	const FMatrix4 viewMatrix = camera.GetViewMatrix();
	const FMatrix4& projectionMatrix = camera.GetProjectionMatrix();
	const FMatrix4 viewProjectionMatrix = projectionMatrix * viewMatrix;

	UpdateCameraData(camPos, camFwd);

#ifdef BENCHMARK
	float VertexShadingMs{};
	float TriangleAssemblingMs{};
#ifdef BINNING
	float TriangleBinningMs{};
#endif
	float RasterizationMs{};
	float PixelShadingMs{};
#endif

	cudaStream_t binnerStream;
	CheckErrorCuda(cudaStreamCreate(&binnerStream));

	m_TotalVisibleNumTriangles = 0;

	const std::vector<CUDAMesh*>& pCudaMeshes = scene.GetCUDAMeshes();
	for (CUDAMesh* pCudaMesh : pCudaMeshes)
	{
		//Mesh Data
		const FMatrix4& worldMat = pCudaMesh->GetWorldConst();
		const FMatrix3& rotationMatrix = pCudaMesh->GetRotationMatrix();
		const FMatrix4 worldViewProjectionMatrix = viewProjectionMatrix * worldMat;

		//Update const data
		UpdateWorldMatrixData(worldMat, worldViewProjectionMatrix, rotationMatrix);
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
		CheckErrorCuda(cudaMemset(m_Dev_NumVisibleTriangles, 0, sizeof(unsigned int)));
		int binnerStatus = 0;
		CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, &binnerStatus, sizeof(int), IsFinishedBinningIdx * 4));

		//---STAGE 2---:  Perform Triangle Assembling
		TriangleAssembler(pCudaMesh, camFwd, cm);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 2---

		unsigned int numVisibleTriangles;
		CheckErrorCuda(cudaMemcpy(&numVisibleTriangles, m_Dev_NumVisibleTriangles, 4, cudaMemcpyDeviceToHost));
		m_TotalVisibleNumTriangles += numVisibleTriangles;

#ifdef BENCHMARK
		TriangleAssemblingMs += StopTimer();
#endif

		//TODO: too many kernel launches
		//persistent kernel approach + global atomic flag value set in host
		//TODO: can enable UNSAFE execution of GPU Atomic Queues due to the TB and RA kernels only processing BinQueueMaxSize amount of triangles per launch
		const unsigned int queueMaxSize = m_BinQueues.GetQueueMaxSize();
		const unsigned int numLoops = (numVisibleTriangles + (queueMaxSize - 1)) / queueMaxSize;
		for (unsigned int i{}; i < numLoops; ++i)
		{

#ifdef BINNING

#ifdef BENCHMARK
			StartTimer();
#endif

			//---STAGE 3---:  Perform Output Vertex Assembling
			TriangleBinner(pCudaMesh, numVisibleTriangles, i * queueMaxSize);
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

			m_BinQueues.ResetQueueSizes();

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
	PixelShader(sampleState, visualisationState);
	CheckErrorCuda(cudaDeviceSynchronize());

	CheckErrorCuda(cudaStreamDestroy(binnerStream));

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
void CUDARenderer::RenderAuto(const CUDASceneGraph& scene, const CUDATextureManager& tm, const Camera& camera)
{
#ifdef _DEBUG
	if (EnterValidRenderingState())
		exit(1);
#else
	EnterValidRenderingState();
#endif

	Render(scene, tm, camera);

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
	const size_t size = m_WindowHelper.Resolution.Width * m_WindowHelper.Resolution.Height * sizeof(unsigned int);
	//We can directly read/write from pixelbuffer
	CheckErrorCuda(cudaMemcpy(m_WindowHelper.pBackBufferPixels, m_CUDAWindowHelper.GetDev_FrameBuffer(), size, cudaMemcpyDeviceToHost));
	//Release a surface after directly accessing the pixels.
	SDL_UnlockSurface(m_WindowHelper.pBackBuffer);
	//Copy the window surface to the screen.
	SDL_BlitSurface(m_WindowHelper.pBackBuffer, 0, m_WindowHelper.pFrontBuffer, 0);
	//Update Window's surface
	SDL_UpdateWindowSurface(m_WindowHelper.pWindow);
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
	unsigned int size{};
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

	m_CUDAWindowHelper.Init(width, height);

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

	size = sizeof(unsigned int);
	cudaFree(m_Dev_NumVisibleTriangles);
	cudaMalloc((void**)&m_Dev_NumVisibleTriangles, size);
	cudaMemset(m_Dev_NumVisibleTriangles, 0, size);
}

CPU_CALLABLE
void CUDARenderer::FreeCUDADeviceBuffers()
{
	m_BinQueues.Destroy();

	CheckErrorCuda(cudaFree(m_Dev_NumVisibleTriangles));
	m_Dev_NumVisibleTriangles = nullptr;

	m_CUDAWindowHelper.Destroy();
}

CPU_CALLABLE
void CUDARenderer::UpdateCameraData(const FPoint3& camPos, const FVector3& camFwd)
{
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, camPos.data, sizeof(camPos), CamPosIdx * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, camFwd.data, sizeof(camFwd), CamFwdIdx * 4));
}

CPU_CALLABLE
void CUDARenderer::UpdateWorldMatrixData(const FMatrix4& worldMatrix, const FMatrix4& wvpMat, const FMatrix3& rotationMat)
{
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, worldMatrix.data, sizeof(worldMatrix), WorldMatIdx * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, wvpMat.data, sizeof(wvpMat), WVPMatIdx * 4));
	CheckErrorCuda(cudaMemcpyToSymbol(dev_ConstMemory, rotationMat.data, sizeof(rotationMat), RotMatIdx * 4));
}

CPU_CALLABLE
cudaError_t CUDARenderer::WaitForStream(cudaStream_t stream)
{
	return cudaStreamSynchronize(stream);
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
void CLEAR_DepthBufferKernel(int* dev_DepthBuffer, int value, const unsigned int width, const unsigned int height)
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
void CLEAR_FrameBufferKernel(unsigned int* dev_FrameBuffer, const unsigned int width, const unsigned int height, unsigned int colour32)
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
void CLEAR_SetPixelShadeBufferKernel(PixelShade* dev_PixelShadeBuffer, const unsigned int width, const unsigned int height, unsigned int colour32)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_PixelShadeBuffer[pixelIdx].colour32 = colour32;
	}
}

GPU_KERNEL
void CLEAR_PixelShadeBufferKernel(PixelShade* dev_PixelShadeBuffer, const unsigned int sizeInWORDs)
{
	//every thread sets 1 WORD of data
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < sizeInWORDs)
	{
		reinterpret_cast<float*>(dev_PixelShadeBuffer)[idx] = 0.f;
	}
}

GPU_KERNEL
void CLEAR_DepthMutexBufferKernel(int* dev_MutexBuffer, const unsigned int width, const unsigned int height)
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
void CLEAR_BuffersKernel(unsigned int* dev_FrameBuffer, int* dev_DepthBuffer, int value, const unsigned int width, const unsigned int height, unsigned int colour32)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_FrameBuffer[pixelIdx] = colour32;
		//__syncthreads(); //for synchronised coalesced global access
		dev_DepthBuffer[pixelIdx] = value;
	}
}

#pragma endregion

GPU_KERNEL
void VS_Kernel(const IVertex* __restrict__ dev_IVertices, OVertex* dev_OVertices, unsigned int numVertices)
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

	//TODO: const device matrices not updating after init???
	//???????????????????????????????????????????????????????????????????????????????????????
	//struct memory is unexpected for some reason, it used to work before ;.;
	//apparently the const memory should be accessed "sequentially"? by each individual thread
	const FPoint3& camPos = reinterpret_cast<const FPoint3&>(dev_ConstMemory[CamPosIdx]);
	const FMatrix4& worldMatrix = reinterpret_cast<const FMatrix4&>(dev_ConstMemory[WorldMatIdx]);
	const FMatrix4& WVPMatrix = reinterpret_cast<const FMatrix4&>(dev_ConstMemory[WVPMatIdx]);
	const FMatrix3& rotationMatrix = reinterpret_cast<const FMatrix3&>(dev_ConstMemory[RotMatIdx]);

	const unsigned int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexIdx < numVertices)
	{
		const IVertex& iVertex = dev_IVertices[vertexIdx];
		OVertex oVertex = GetNDCVertex(iVertex, WVPMatrix, worldMatrix, rotationMatrix, camPos);
		dev_OVertices[vertexIdx] = oVertex;
	}
}

GPU_KERNEL
void TA_Kernel(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer, unsigned int numIndices,
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
void RA_PerTriangleKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices, 
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
void PS_Kernel(unsigned int* dev_FrameBuffer, const PixelShade* __restrict__ const dev_PixelShadeBuffer,
	SampleState sampleState, const unsigned int width, const unsigned int height)
{
	//Notes: PixelShade has size of 32, but bank conflicts
	//TODO: store PixelShade data column-based to avoid bank conflicts, but faster access?
	//GPU_SHARED_MEMORY PixelShade pixelShadeSharedMemory[width * height];

	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	//TODO: if-statement is not necessary for standard resolutions
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		const PixelShade pixelShade = dev_PixelShadeBuffer[pixelIdx];
		const CUDATexturesCompact& textures = pixelShade.textures;
		if (textures.Diff.dev_pTex != 0)
		{
			RGBColor colour = ShadePixelSafe(textures, pixelShade.uv, pixelShade.n, pixelShade.tan, pixelShade.vd, sampleState);
			RGBA rgba = RGBA::GetRGBAFromColour(colour);
			dev_FrameBuffer[pixelIdx] = rgba.colour32;
		}
		else
		{
			dev_FrameBuffer[pixelIdx] = pixelShade.colour32;
		}
	}
}

GPU_KERNEL
void PS_VisualiseDepthColourKernel(unsigned int* dev_FrameBuffer, const PixelShade* __restrict__ const dev_PixelShadeBuffer,
	const unsigned int width, const unsigned int height)
{
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	//TODO: if-statement is not necessary
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		const float zInterpolated = dev_PixelShadeBuffer[pixelIdx].zInterpolated;
		RGBA rgba;
		rgba.r8 = 0; //For SDL: R and B values are swapped
		rgba.g8 = 0;
		rgba.b8 = (unsigned char)(Remap(zInterpolated, 0.985f, 1.f) * 255);
		rgba.a8 = 0;
		dev_FrameBuffer[pixelIdx] = rgba.colour32;
	}
}

GPU_KERNEL
void PS_VisualiseNormalKernel(unsigned int* dev_FrameBuffer, const PixelShade* __restrict__ const dev_PixelShadeBuffer,
	const unsigned int width, const unsigned int height)
{
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	//TODO: if-statement is not necessary
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		const FVector3 normal = dev_PixelShadeBuffer[pixelIdx].n;
		const RGBColor& colour = reinterpret_cast<const RGBColor&>(normal); //able to do this, since they're both POD structs
		RGBA rgba = RGBA::GetRGBAFromColour(colour);
		dev_FrameBuffer[pixelIdx] = rgba.colour32;
	}
}

#pragma region Binning

//TODO: templated kernel, for bin queues of datatype T (unsigned int TriangleIdx)

GPU_KERNEL
void TB_Kernel(TriangleIdx* dev_Triangles, unsigned int numVisibleTriangles, unsigned int triangleIdxOffset,
	unsigned int* dev_BinQueueSizes, unsigned int* dev_BinQueues, int* dev_BinQueueSizesMutexBuffer,
	const OVertex* dev_OVertices, 
	IPoint2 numBins, IPoint2 binDim, unsigned int binQueueMaxSize, unsigned int width, unsigned int height)
{
	//TODO: use shared memory to copy faster
	//data size of 9 shows no bank conflicts!
	//TriangleIdx can stay in local memory (registers)
	//DEPENDS ON REGISTER USAGE

	//IDEA: (concept taken from Laine et al.)
	//Shared memory as a list of bits, 48 bytes per triangle(/thread), which covers 384 bins (== bits)
	//This will maximize occupancy for any amount of threads per block
	//Less amount of bins is applicable too

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

		for (int x{ binMinX }; x <= binMaxX; ++x)
		{
			for (int y{ binMinY }; y <= binMaxY; ++y)
			{
				//atomically add triangle to bin queue
				const unsigned int binIdx = x + y * numBins.x;

				//Deliberately not using a helper struct, since it would waste (note: A LOT of) memory and operations
				CUDAAtomicQueueOP::Insert(&dev_BinQueues[binIdx * binQueueMaxSize],
					dev_BinQueueSizes[binIdx],
					dev_BinQueueSizesMutexBuffer[binIdx],
					binQueueMaxSize,
					triangleIdx);

				//bool isDone = false;
				//do
				//{
				//	isDone = (atomicCAS(&dev_BinQueueSizesMutexBuffer[binIdx], 0, 1) == 0);
				//	if (isDone)
				//	{
				//		//critical section
				//		const unsigned int currQueueSize = dev_BinQueueSizes[binIdx];
				//
				//		if (currQueueSize < binQueueMaxSize)
				//		{
				//			//insert triangle Idx in queue
				//			dev_BinQueues[binIdx * binQueueMaxSize + currQueueSize] = triangleIdx;
				//			//increase bin's queue size
				//			++dev_BinQueueSizes[binIdx];
				//
				//			//dev_BinQueueSizesMutexBuffer[binIdx] = 0; //release lock
				//		}
				//		dev_BinQueueSizesMutexBuffer[binIdx] = 0; //release lock
				//		//end of critical section
				//	}
				//} while (!isDone);
			}
		}
	}
}

GPU_KERNEL
void RA_PerBinKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices,
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
void RA_PerTileKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, int* dev_DepthMutexBuffer, CUDATexturesCompact textures,
	unsigned int* dev_BinQueues, unsigned int* dev_BinQueueSizes, int* dev_BinQueueSizesMutexBuffer, IPoint2 binDim, unsigned int binQueueMaxSize,
	const unsigned int pixelCoverageX, const unsigned int pixelCoverageY, const unsigned int width, const unsigned int height)
{
	//TODO: blocksize.x == binQueueMaxSize

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

#ifdef FINERASTER_SHAREDMEM
	//Occupancy stays higher due to less shared memory usage/block
	//CONCERN: occupancy is already low due to high register usage
	//Perhaps use this big amount of shared memory paired with the higher register usage to "mask" them

	extern GPU_SHARED_MEMORY float positionsBuffer[];

	if (threadIdx.x < queueSize)
	{
		const unsigned int binQueueIdx = dev_BinQueues[binIdx * binQueueMaxSize + threadIdx.x];
		const TriangleIdx triangle = dev_Triangles[binQueueIdx];

		const OVertex& v0 = dev_OVertices[triangle.idx0];
		const OVertex& v1 = dev_OVertices[triangle.idx1];
		const OVertex& v2 = dev_OVertices[triangle.idx2];

		//each thread stores 3 FPoint4's into shared memory at once
		//Only then copy OVertex_Data when pixel is in triangle
		//This will actually maximize the occupancy of shared memory usage for every thread block
		FPoint4* pPositions = reinterpret_cast<FPoint4*>(positionsBuffer);

		//Disadvantage: 2 global memory accesses
		//Advantage: fast copy of all Positions into shared memory and faster access
		OVertex_PosShared test0{};
		OVertex_PosShared test1{};
		OVertex_PosShared test2{};
		test0.pPos = &pPositions[threadIdx.x];
		test1.pPos = &pPositions[threadIdx.x + 1];
		test2.pPos = &pPositions[threadIdx.x + 2];

		//copy to shared memory
		*test0.pPos = v0.p;
		*test1.pPos = v1.p;
		*test2.pPos = v2.p;

		//copy to local memory
		memcpy(&test0.vData, &v0 + sizeof(FPoint4), sizeof(OVertexData));
		memcpy(&test1.vData, &v1 + sizeof(FPoint4), sizeof(OVertexData));
		memcpy(&test2.vData, &v2 + sizeof(FPoint4), sizeof(OVertexData));

		//acts as a barrier for shared memory
		__syncthreads();

		NDCToScreenSpace(test0.pPos->xy, test1.pPos->xy, test2.pPos->xy, width, height);
		RasterizeTriangle(bb, v0, v1, v2, dev_DepthMutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);
	}
#else
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
	}
#endif
}

#pragma endregion

#pragma region DEPRECATED

GPU_KERNEL
void VS_PrototypeKernel_DEP(const IVertex* __restrict__ dev_IVertices, OVertex* dev_OVertices, unsigned int numVertices)
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
void TA_Kernel_DEP(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer,
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
				triangle.idx1 = triangle.idx1 + triangle.idx2;
				triangle.idx2 = triangle.idx1 - triangle.idx2;
				triangle.idx1 = triangle.idx1 - triangle.idx2;
			}
			memcpy(&dev_Triangles[indexIdx], &triangle, sizeof(TriangleIdx));
		}
	}
}

GPU_KERNEL
void RA_PerPixelKernel_DEP(const TriangleIdx* __restrict__ dev_Triangles, const OVertex* __restrict__ const dev_OVertices, unsigned int numTriangles,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, CUDATexturesCompact textures,
	const unsigned int width, const unsigned int height)
{
	//TODO: each thread represents a pixel
	//each thread loops through all triangles
	//triangles are stored in shared memory (broadcast)
	//advantage: thread only does 1 check per triangle w/o looping for all pixels 
	//=> O(n) n = numTriangles vs O(n^m) n = numTriangles m = numPixels
	//advantage: nomore atomic operations needed bc only 1 thread can write to 1 unique pixelIdx

	constexpr float* pCamFwd = &dev_ConstMemory[CamFwdIdx];

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

		if (!IsTriangleVisible(v0.p.xyz, v1.p.xyz, v2.p.xyz))
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
void RA_PerTriangleKernel_DEP(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices, unsigned int numTriangles,
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

	if (!IsTriangleVisible(v0.p.xyz, v1.p.xyz, v2.p.xyz))
	{
		return;
	}

	NDCToScreenSpace(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
	const BoundingBox bb = GetBoundingBox(v0.p.xy, v1.p.xy, v2.p.xy, width, height);
	//Rasterize Screenspace triangle
	RasterizeTriangle(bb, v0, v1, v2, dev_MutexBuffer, dev_DepthBuffer, dev_PixelShadeBuffer, width, textures);
}

GPU_KERNEL
void TA_TB_Kernel_DEP(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer, unsigned int numIndices,
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

			//Deliberately not using a helper struct, since it would waste (note: A LOT of) memory and operations
			CUDAAtomicQueueOP::Insert(&dev_BinQueues[binIdx * binQueueMaxSize],
				dev_BinQueueSizes[binIdx],
				dev_BinQueueSizesMutexBuffer[binIdx],
				binQueueMaxSize,
				triangleIdx);

			//bool isDone = false;
			//do
			//{
			//	isDone = (atomicCAS(&dev_BinQueueSizesMutexBuffer[binIdx], 0, 1) == 0);
			//	if (isDone)
			//	{
			//		//critical section
			//		const unsigned int currQueueSize = dev_BinQueueSizes[binIdx];
			//		if (currQueueSize < binQueueMaxSize)
			//		{
			//			//insert triangle Idx in queue
			//			dev_BinQueues[binIdx * binQueueMaxSize + currQueueSize] = triangleIdx;
			//			//increase bin's queue size
			//			++dev_BinQueueSizes[binIdx];
			//		}
			//		dev_BinQueueSizesMutexBuffer[binIdx] = 0;
			//		//end of critical section
			//	}
			//} while (!isDone);
		}
	}
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

	//WINDOW BUFFER DATA
	unsigned int* dev_FrameBuffer = m_CUDAWindowHelper.GetDev_FrameBuffer();
	int* dev_DepthBuffer = m_CUDAWindowHelper.GetDev_DepthBuffer();
	int* dev_DepthMutexBuffer = m_CUDAWindowHelper.GetDev_DepthMutexBuffer();
	PixelShade* dev_PixelShadeBuffer = m_CUDAWindowHelper.GetDev_FragmentBuffer();

	//NO CONCURRENCY
	
	{
		constexpr int depthBufferResetValue = INT_MAX;
		const dim3 numThreadsPerBlock{ 32, 32 };
		const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
		CLEAR_DepthBufferKernel << <numBlocks, numThreadsPerBlock>>>
			(dev_DepthBuffer, depthBufferResetValue, width, height);
	}
	{
		const size_t sizeInWords = width * height * (sizeof(PixelShade) / 4);
		constexpr unsigned int numThreadsPerBlock = 512;
		const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
		CLEAR_PixelShadeBufferKernel << <numBlocks, numThreadsPerBlock>>>
			(dev_PixelShadeBuffer, sizeInWords);
	}
	{
		const RGBA rgba{ colour };
		const dim3 numThreadsPerBlock{ 32, 32 };
		const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
		CLEAR_SetPixelShadeBufferKernel << <numBlocks, numThreadsPerBlock>>>
			(dev_PixelShadeBuffer, width, height, rgba.colour32);
	}
	//{
	//	////UNNECESSARY STEP: mutexbuffer should always revert to base initialized state, otherwise deadlocks would occur
	//	//const dim3 numThreadsPerBlock{ 32, 32 };
	//	//const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
	//	//ClearDepthMutexBufferKernel<<<numBlocks, numThreadsPerBlock>>>
	//	//	(dev_DepthMutexBuffer, width, height);
	//}

	//CONCURRENCY TEST
	
	//cudaStream_t depthBufferStream{}, pixelShadeBufferStream{};
	//CheckErrorCuda(cudaStreamCreate(&depthBufferStream));
	//CheckErrorCuda(cudaStreamCreate(&pixelShadeBufferStream));
	//{
	//	constexpr int depthBufferResetValue = INT_MAX;
	//	const dim3 numThreadsPerBlock{ 32, 32 };
	//	const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
	//	ClearDepthBufferKernel<<<numBlocks, numThreadsPerBlock, 0, depthBufferStream>>>
	//		(dev_DepthBuffer, depthBufferResetValue, width, height);
	//}
	//{
	//	const size_t sizeInWords = width * height * (sizeof(PixelShade) / 4);
	//	constexpr unsigned int numThreadsPerBlock = 512;
	//	const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
	//	ClearPixelShadeBufferKernel<<<numBlocks, numThreadsPerBlock, 0, pixelShadeBufferStream>>>
	//		(dev_PixelShadeBuffer, sizeInWords);
	//}
	//{
	//	CheckErrorCuda(WaitForStream(pixelShadeBufferStream)); //wait for other kernel to finish to prevent data races
	//	//UNNECESSARY STEP: pixelshade stage will overwrite this anyway + more kernel launch overhead
	//	const RGBA rgba{ colour };
	//	const dim3 numThreadsPerBlock{ 32, 32 };
	//	const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
	//	//Needs to be called after ClearPixelShadeBufferKernel
	//	SetPixelShadeBufferKernel<<<numBlocks, numThreadsPerBlock, 0, pixelShadeBufferStream>>>
	//		(dev_PixelShadeBuffer, width, height, rgba.colour32);
	//	////Not necessary, since we overwrite the entire buffer every frame anyway
	//	//ClearFrameBufferKernel<<<numBlocks, numThreadsPerBlock>>>
	//	//	(dev_FrameBuffer, width, height, rgba.colour32);
	//}
	//CheckErrorCuda(WaitForStream(depthBufferStream));
	//CheckErrorCuda(cudaStreamDestroy(pixelShadeBufferStream));
	//CheckErrorCuda(cudaStreamDestroy(depthBufferStream));

	//MERGED CLEARING KERNEL TEST

	//{
	//	const RGBA rgba{ colour };
	//	constexpr int depthBufferResetValue = INT_MAX;
	//	const dim3 numThreadsPerBlock{ 32, 32 };
	//	const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };
	//	ClearBuffersKernel<<<numBlocks, numThreadsPerBlock>>>
	//		(dev_FrameBuffer, dev_DepthMutexBuffer, depthBufferResetValue, width, height, rgba.colour32);
	//}
	//{
	//	const size_t sizeInWords = width * height * (sizeof(PixelShade) / 4);
	//	constexpr unsigned int numThreadsPerBlock = 512;
	//	const unsigned int numBlocks = (unsigned int)(sizeInWords + numThreadsPerBlock - 1) / numThreadsPerBlock;
	//	ClearPixelShadeBufferKernel<<<numBlocks, numThreadsPerBlock>>>
	//		(dev_PixelShadeBuffer, sizeInWords);
	//}
}

CPU_CALLABLE
void CUDARenderer::VertexShader(const CUDAMesh* pCudaMesh)
{
	const unsigned int numVertices = pCudaMesh->GetNumVertices();

	const IVertex* dev_IVertexBuffer = pCudaMesh->GetDevIVertexBuffer();
	OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();

	//constexpr unsigned int paddedSizeOfIVertex = sizeof(OVertex) / 4 + 1;
	//constexpr unsigned int sharedMemoryNeededPerThread = paddedSizeOfIVertex * 4;
	//constexpr unsigned int sharedMemoryNeededPerWarp = sharedMemoryNeededPerThread * 32;
	constexpr unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (numVertices + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	//const unsigned int numSharedMemory = numThreadsPerBlock * sharedMemoryNeededPerThread;
	VS_Kernel<<<numBlocks, numThreadsPerBlock>>>(
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

	//MESH DATA
	const PrimitiveTopology topology = pCudaMesh->GetTopology();
	const unsigned int numIndices = pCudaMesh->GetNumIndices();
	const unsigned int numTriangles = pCudaMesh->GetTotalNumTriangles();

	const unsigned int* dev_IndexBuffer = pCudaMesh->GetDevIndexBuffer();
	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

	//KERNEL LAUNCH PARAMS
	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (numTriangles + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	TA_Kernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, dev_IndexBuffer, numIndices, m_Dev_NumVisibleTriangles,
		dev_OVertexBuffer, topology, cm, camFwd,
		width, height);

	//TriangleAssemblerKernelOld<<<numBlocks, numThreadsPerBlock>>>(
	//	dev_Triangles[mi.Idx], dev_IndexBuffer[mi.Idx], 
	//	numIndices, topology);
}

CPU_CALLABLE
void CUDARenderer::TriangleBinner(const CUDAMesh* pCudaMesh, const unsigned int numVisibleTriangles, const unsigned int triangleIdxOffset, cudaStream_t stream)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

	//MESH DATA
	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

	//BINNING DATA
	unsigned int binQueuesMaxSize = m_BinQueues.GetQueueMaxSize();
	const IPoint2 numBins = { (int)m_BinQueues.GetNumQueuesX(), (int)m_BinQueues.GetNumQueuesY() };
	unsigned int* dev_BinQueueSizes = m_BinQueues.GetDev_QueueSizes();
	unsigned int* dev_BinQueues = m_BinQueues.GetDev_Queues();
	int* dev_BinQueueSizesMutexBuffer = m_BinQueues.GetDev_QueueSizesMutexBuffer();

	//KERNEL LAUNCH PARAMS
	const unsigned int numThreadsPerBlock = m_BinQueues.GetQueueMaxSize();
	const unsigned int numBlocks = 1;
	TB_Kernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, numVisibleTriangles, triangleIdxOffset,
		dev_BinQueueSizes, dev_BinQueues, dev_BinQueueSizesMutexBuffer,
		dev_OVertexBuffer,
		numBins, m_BinDim, binQueuesMaxSize, width, height);
}

CPU_CALLABLE
void CUDARenderer::TriangleAssemblerAndBinner(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm, cudaStream_t stream)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

	//WINDOW BUFFER DATA
	unsigned int* dev_FrameBuffer = m_CUDAWindowHelper.GetDev_FrameBuffer();
	PixelShade* dev_PixelShadeBuffer = m_CUDAWindowHelper.GetDev_FragmentBuffer();
	int* dev_DepthBuffer = m_CUDAWindowHelper.GetDev_DepthBuffer();
	int* dev_DepthMutexBuffer = m_CUDAWindowHelper.GetDev_DepthMutexBuffer();

	//MESH DATA
	const unsigned int numIndices = pCudaMesh->GetNumIndices();
	const PrimitiveTopology topology = pCudaMesh->GetTopology();
	const unsigned int numTriangles = pCudaMesh->GetTotalNumTriangles();

	const unsigned int* dev_IndexBuffer = pCudaMesh->GetDevIndexBuffer();
	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

	//BINNING DATA
	unsigned int binQueueMaxSize = m_BinQueues.GetQueueMaxSize();
	const IPoint2 numBins = { (int)m_BinQueues.GetNumQueuesX(), (int)m_BinQueues.GetNumQueuesY() };
	unsigned int* dev_BinQueueSizes = m_BinQueues.GetDev_QueueSizes();
	unsigned int* dev_BinQueues = m_BinQueues.GetDev_Queues();
	int* dev_BinQueueSizesMutexBuffer = m_BinQueues.GetDev_QueueSizesMutexBuffer();

	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (numTriangles + (numThreadsPerBlock - 1)) / numThreadsPerBlock;
	TA_TB_Kernel_DEP<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, dev_IndexBuffer, numIndices, m_Dev_NumVisibleTriangles,
		dev_BinQueueSizes, dev_BinQueues, dev_BinQueueSizesMutexBuffer,
		dev_OVertexBuffer, topology, cm, camFwd,
		numBins, m_BinDim, binQueueMaxSize, width, height);
}

CPU_CALLABLE
void CUDARenderer::Rasterizer(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm, cudaStream_t stream)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

	//WINDOW BUFFER DATA
	unsigned int* dev_FrameBuffer = m_CUDAWindowHelper.GetDev_FrameBuffer();
	PixelShade* dev_PixelShadeBuffer = m_CUDAWindowHelper.GetDev_FragmentBuffer();
	int* dev_DepthBuffer = m_CUDAWindowHelper.GetDev_DepthBuffer();
	int* dev_DepthMutexBuffer = m_CUDAWindowHelper.GetDev_DepthMutexBuffer();

	//MESH DATA
	const CUDATexturesCompact& textures = pCudaMesh->GetTextures();
	const OVertex* dev_OVertexBuffer = pCudaMesh->GetDevOVertexBuffer();
	TriangleIdx* dev_TriangleBuffer = pCudaMesh->GetDevTriangleBuffer();

	//BINNING DATA
	unsigned int binQueuesMaxSize = m_BinQueues.GetQueueMaxSize();
	const IPoint2 numBins = { (int)m_BinQueues.GetNumQueuesX(), (int)m_BinQueues.GetNumQueuesY() };
	unsigned int* dev_BinQueueSizes = m_BinQueues.GetDev_QueueSizes();
	unsigned int* dev_BinQueues = m_BinQueues.GetDev_Queues();
	int* dev_BinQueueSizesMutexBuffer = m_BinQueues.GetDev_QueueSizesMutexBuffer();

#ifdef BINNING
#ifdef FINERASTER
	const dim3 numThreadsPerBlock = { 16, 16 };
	const dim3 numBlocks = { m_BinQueues.GetNumQueuesX(), m_BinQueues.GetNumQueuesY() };
	//const unsigned int numSharedMemory = m_BinQueues.QueueMaxSize * 4 + 4; //queue array + 1 queue size

	//pixel coverage per thread
	const unsigned int pixelCoverageX = m_BinDim.x / numThreadsPerBlock.x;
	const unsigned int pixelCoverageY = m_BinDim.y / numThreadsPerBlock.y;
	RA_PerTileKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, dev_OVertexBuffer,
		dev_PixelShadeBuffer, dev_DepthBuffer, dev_DepthMutexBuffer, textures,
		dev_BinQueues, dev_BinQueueSizes, dev_BinQueueSizesMutexBuffer, m_BinDim,
		binQueuesMaxSize, pixelCoverageX, pixelCoverageY, width, height);
	//TODO: each block iterates through entire queue array, 
	//each block should do 1 triangle of 1 queue
#else
	constexpr unsigned int numThreadsPerBlock = 256;
	const dim3 numBlocks = { m_BinQueues.GetNumQueuesX(), m_BinQueues.GetNumQueuesY() };
	RasterizerPerBinKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
		dev_TriangleBuffer, dev_OVertexBuffer[mi.Idx],
		dev_PixelShadeBuffer, dev_DepthBuffer, dev_DepthMutexBuffer, textures,
		dev_BinQueues, dev_BinQueueSizes, dev_BinQueueSizesMutexBuffer, m_BinDim,
		m_BinQueues.GetQueueMaxSize(), width, height);
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
void CUDARenderer::PixelShader(SampleState sampleState, VisualisationState visualisationState)
{
	const unsigned int width = m_WindowHelper.Resolution.Width;
	const unsigned int height = m_WindowHelper.Resolution.Height;

	unsigned int* dev_FrameBuffer = m_CUDAWindowHelper.GetDev_FrameBuffer();
	PixelShade* dev_PixelShadeBuffer = m_CUDAWindowHelper.GetDev_FragmentBuffer();

	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ width / numThreadsPerBlock.x, height / numThreadsPerBlock.y };

	switch (visualisationState)
	{
	case VisualisationState::PBR:
		PS_Kernel<<<numBlocks, numThreadsPerBlock>>>(dev_FrameBuffer, dev_PixelShadeBuffer, sampleState, width, height);
		break;
	case VisualisationState::Depth:
		PS_VisualiseDepthColourKernel<<<numBlocks, numThreadsPerBlock>>>(dev_FrameBuffer, dev_PixelShadeBuffer, width, height);
		break;
	case VisualisationState::Normal:
		PS_VisualiseNormalKernel<<<numBlocks, numThreadsPerBlock>>>(dev_FrameBuffer, dev_PixelShadeBuffer, width, height);
		break;
	default:
		break;
	}
}

CPU_CALLABLE
void CUDARenderer::DrawTexture(char* tP)
{
	CUDATexture texture{};
	texture.Create(tP);

	EnterValidRenderingState();

	//WINDOW BUFFER DATA
	unsigned int* dev_FrameBuffer = m_CUDAWindowHelper.GetDev_FrameBuffer();

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

	//WINDOW BUFFER DATA
	unsigned int* dev_FrameBuffer = m_CUDAWindowHelper.GetDev_FrameBuffer();

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
void CUDARenderer::KernelWarmUp()
{
	CLEAR_DepthBufferKernel<<<0, 0>>>(nullptr, 0, 0, 0);
	CLEAR_FrameBufferKernel<<<0, 0>>>(nullptr, 0, 0, 0);
	CLEAR_PixelShadeBufferKernel<<<0, 0>>>(nullptr, 0);
	CLEAR_DepthMutexBufferKernel <<<0, 0>>>(nullptr, 0, 0);
	VS_Kernel<<<0, 0>>>(nullptr, nullptr, 0);
	TA_Kernel<<<0, 0>>>(nullptr, nullptr, 0, nullptr, nullptr, (PrimitiveTopology)0, (CullingMode)0, {}, 0, 0);
	RA_PerTriangleKernel<<<0, 0>>>(nullptr, nullptr, 0, nullptr, nullptr, nullptr, {}, 0, 0);
	PS_Kernel<<<0, 0>>>(nullptr, nullptr, SampleState(0), 0, 0);
}

#pragma endregion