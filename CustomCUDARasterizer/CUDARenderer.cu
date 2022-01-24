#include "PCH.h"
#include "CUDARenderer.cuh"
#include <vector>

//Project CUDA includes
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
#include "RGBRaw.h"

#pragma region STRUCT DECLARATIONS

struct RenderData
{
	FPoint3 camPos;
	FMatrix4 viewProjectionMatrix;
	FMatrix4 worldMatrix;
};

union RenderDataRaw
{
	//compiler BUG: attempting to reference a deleted function?
	float* data;
	RenderData renderData;
};

struct Triangle
{
	OVertex v0;
	OVertex v1;
	OVertex v2;
};

#pragma endregion

#pragma region GLOBAL VARIABLES

//CONST DEVICE MEMORY - Does NOT have to be allocated or freed
GPU_CONST_MEMORY static
float dev_RenderData_const[sizeof(RenderData) / sizeof(float)]{};
//NOTE: cannot contain anything else besides primitive variables (int, float, etc.)

//DEVICE MEMORY - Does have to be allocated and freed
static unsigned int* dev_FrameBuffer{};
static float* dev_DepthBuffer{};
static Triangle* dev_Triangles{};
static int* dev_Mutex{};
static std::vector<IVertex*> dev_IVertexBuffer{};
static std::vector<unsigned int*> dev_IndexBuffer{};
static std::vector<OVertex*> dev_OVertexBuffer{};

#pragma endregion

//TODO: to counter global memory access sequencing
//allocate buffers/dynamic pools per BLOCK (per screen bin)
//Same goes with shared memory, but has bank conflicts

//--------------------------

CPU_CALLABLE
CUDARenderer::CUDARenderer(const WindowHelper& windowHelper)
	: m_WindowHelper{ windowHelper }
	, m_NumTriangles{}
	, m_h_pFrameBuffer{}
	, m_MeshIdentifiers{}
{
	InitCUDARasterizer();

	CheckErrorCuda(cudaEventCreate(&m_StartEvent));
	CheckErrorCuda(cudaEventCreate(&m_StopEvent));
}

CPU_CALLABLE
CUDARenderer::~CUDARenderer()
{
	CheckErrorCuda(DeviceSynchroniseCuda());
	CheckErrorCuda(cudaEventDestroy(m_StartEvent));
	CheckErrorCuda(cudaEventDestroy(m_StopEvent));
	FreeCUDARasterizer();
}

#pragma region MISC HELPER FUNCTIONS

CPU_CALLABLE static
std::string ToKbs(size_t bytes)
{
	const size_t toKbs = 1024;
	std::string output{ std::to_string(bytes / toKbs) + "Kb" };
	return output;
}

CPU_CALLABLE static
std::string ToMbs(size_t bytes)
{
	const size_t toMBs = 1024 * 1024;
	std::string output{ std::to_string(bytes / toMBs) + "Mb" };
	return output;
}

CPU_CALLABLE static
std::string ToGbs(size_t bytes)
{
	const size_t toGBs = 1024 * 1024 * 1024;
	std::string output{ std::to_string(bytes / toGBs) + "Gb" };
	return output;
}

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
void CUDARenderer::InitCUDARasterizer()
{
#ifdef _DEBUG
	DisplayGPUSpecs(0);
#endif

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

	//The framebuffer in device memory
	size = sizeof(unsigned int);
	CheckErrorCuda(cudaFree(dev_FrameBuffer));
	CheckErrorCuda(cudaMalloc((void**)&dev_FrameBuffer, width * height * size));
	CheckErrorCuda(cudaMemset(dev_FrameBuffer, 0, width * height * size));

	size = sizeof(float);
	CheckErrorCuda(cudaFree(dev_DepthBuffer));
	CheckErrorCuda(cudaMalloc((void**)&dev_DepthBuffer, width * height * size));
	CheckErrorCuda(cudaMemset(dev_DepthBuffer, 0, width * height * size));

	size = sizeof(int);
	cudaFree(dev_Mutex);
	cudaMalloc((void**)&dev_Mutex, width * height * size);
	cudaMemset(dev_Mutex, 0, width * height * size);

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
void CUDARenderer::AllocateMeshBuffers(const size_t numVertices, const size_t numIndices, size_t meshIdx)
{
	//If no sufficient space in vector, enlarge
	const size_t newSize = meshIdx + 1;
	if (newSize > dev_IVertexBuffer.size())
	{
		dev_IVertexBuffer.resize(newSize);
		dev_IndexBuffer.resize(newSize);
		dev_OVertexBuffer.resize(newSize);
	}

	//Free unwanted memory
	CheckErrorCuda(cudaFree(dev_IVertexBuffer[meshIdx]));
	CheckErrorCuda(cudaFree(dev_IndexBuffer[meshIdx]));
	CheckErrorCuda(cudaFree(dev_OVertexBuffer[meshIdx]));

	//Allocate Input Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_IVertexBuffer[meshIdx], numVertices * sizeof(IVertex)));
	//Allocate Index Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_IndexBuffer[meshIdx], numIndices * sizeof(unsigned int)));
	//Allocate Ouput Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_OVertexBuffer[meshIdx], numVertices * sizeof(OVertex)));
}

CPU_CALLABLE
void CUDARenderer::CopyMeshBuffers(const std::vector<IVertex>& vertexBuffer, const std::vector<unsigned int>& indexBuffer, size_t meshIdx)
{
	//Copy Input Vertex Buffer
	CheckErrorCuda(cudaMemcpy(dev_IVertexBuffer[meshIdx], &vertexBuffer[0], vertexBuffer.size() * sizeof(IVertex), cudaMemcpyHostToDevice));
	//Copy Index Buffer
	CheckErrorCuda(cudaMemcpy(dev_IndexBuffer[meshIdx], &indexBuffer[0], indexBuffer.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
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
	}
	m_MeshIdentifiers.clear();
}

CPU_CALLABLE
void CUDARenderer::FreeCUDARasterizer()
{
	//Free buffers
	CheckErrorCuda(cudaFree(dev_FrameBuffer));
	dev_FrameBuffer = nullptr;

	//CheckErrorCuda(cudaFreeHost(m_WindowHelper.pBackBufferPixels));
	//m_WindowHelper.pBackBufferPixels = nullptr;

	CheckErrorCuda(cudaFreeHost(m_h_pFrameBuffer));
	m_h_pFrameBuffer = nullptr;

	CheckErrorCuda(cudaFree(dev_DepthBuffer));
	dev_DepthBuffer = nullptr;

	cudaFree(dev_Mutex);
	dev_Mutex = nullptr;

	CheckErrorCuda(cudaFree(dev_Triangles));
	dev_Triangles = nullptr;

	FreeMeshBuffers();
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
	SDL_BlitSurface(m_WindowHelper.pBackBuffer, 0, m_WindowHelper.pFrontBuffer, 0);
	SDL_UpdateWindowSurface(m_WindowHelper.pWindow); //Copy the window surface to the screen.
}

#pragma endregion

#pragma region GPU HELPER FUNCTIONS

GPU_CALLABLE static
OVertex GetNDCVertex(const IVertex& __restrict__ iVertex, const FPoint3& camPos,
	const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix)
{
	OVertex oVertex;

	const FPoint3 worldPosition{ worldMatrix * FPoint4{ iVertex.p } };
	const FMatrix4 worldViewProjectionMatrix = viewProjectionMatrix * worldMatrix;
	//const FMatrix3 rotationMatrix = (FMatrix3)worldMatrix;

	new (&oVertex.p) FPoint4{ worldViewProjectionMatrix * FPoint4{ iVertex.p } };
	oVertex.p.x /= oVertex.p.w;
	oVertex.p.y /= oVertex.p.w;
	oVertex.p.z /= oVertex.p.w;

	new (&oVertex.vd) const FVector3{ GetNormalized(worldPosition - camPos) };
	new (&oVertex.n) const FVector3{ (FMatrix3)worldMatrix * iVertex.n };
	new (&oVertex.tan) const FVector3{ (FMatrix3)worldMatrix * iVertex.tan };

	oVertex.uv = iVertex.uv;
	oVertex.c = iVertex.c;

	return oVertex;
}

GPU_CALLABLE static
bool EdgeFunction(const FPoint2& v, const FVector2& edge, const FPoint2& pixel, float& weight)
{
	// counter-clockwise
	const FVector2 vertexToPixel{ pixel - v };
	const float cross = Cross(edge, vertexToPixel);
	weight = cross;
	return cross < 0.f;
}

GPU_CALLABLE static
bool IsPixelInTriangle(const Triangle& triangle, const FPoint2& pixel, float weights[3])
{
	const FPoint2& v0 = triangle.v0.p.xy;
	const FPoint2& v1 = triangle.v1.p.xy;
	const FPoint2& v2 = triangle.v2.p.xy;

	const FVector2 edgeA{ v0 - v1 };
	const FVector2 edgeB{ v1 - v2 };
	const FVector2 edgeC{ v2 - v0 };
	// counter-clockwise

	const float totalArea = Cross(edgeA, edgeC);

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

	if (EdgeFunction(v0, edgeA, pixel, weights[2])) return false;
	if (EdgeFunction(v1, edgeB, pixel, weights[1])) return false;
	if (EdgeFunction(v2, edgeC, pixel, weights[0])) return false;
	weights[0] /= totalArea;
	weights[1] /= totalArea;
	weights[2] /= totalArea;

	return true;
}

GPU_CALLABLE static
bool DepthTest(float dev_DepthBuffer[], int dev_Mutex[], const size_t pixelIdx, float zInterpolated)
{
	//TODO: shared memory
	
	//Update depthbuffer atomically
	bool isDone = false;
	do
	{
		isDone = (atomicCAS(&dev_Mutex[pixelIdx], 0, 1) == 0);
		if (isDone)
		{
			if (zInterpolated < dev_DepthBuffer[pixelIdx]) //DEPTH BUFFER INVERTED INTERPRETATION
			{
				dev_DepthBuffer[pixelIdx] = zInterpolated;
			}
			dev_Mutex[pixelIdx] = 0;
		}
	} while (!isDone);
	return true;
	//atomicCAS
	/*
	//int atomicCAS(int* address, int compare, int val);
	//reads the 16 - bit, 32 - bit or 64 - bit word old located at the address address in global or shared memory, 
	//computes(old == compare ? val : old), and stores the result back to memory at the same address.
	//These three operations are performed in one atomic transaction.The function returns old(Compare And Swap).
	*/
}

GPU_CALLABLE static
bool IsAllXOutsideFrustum(const Triangle& triangle)
{
	return	(triangle.v0.p.x < -1.f && triangle.v1.p.x < -1.f && triangle.v2.p.x < -1.f) ||
			(triangle.v0.p.x > 1.f && triangle.v1.p.x > 1.f && triangle.v2.p.x > 1.f);
}

GPU_CALLABLE static
bool IsAllYOutsideFrustum(const Triangle& triangle)
{
	return	(triangle.v0.p.y < -1.f && triangle.v1.p.y < -1.f && triangle.v2.p.y < -1.f) ||
			(triangle.v0.p.y > 1.f && triangle.v1.p.y > 1.f && triangle.v2.p.y > 1.f);
}

GPU_CALLABLE static
bool IsAllZOutsideFrustum(const Triangle& triangle)
{
	return	(triangle.v0.p.z < 0.f && triangle.v1.p.z < 0.f && triangle.v2.p.z < 0.f) ||
			(triangle.v0.p.z > 1.f && triangle.v1.p.z > 1.f && triangle.v2.p.z > 1.f);
}

GPU_CALLABLE static
bool IsTriangleVisible(const Triangle& triangle)
{
	// Solution to FrustumCulling bug
	//	   if (all x values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all y values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all z values are < 0.f or > 1.f) AT ONCE, cull
	return(!IsAllXOutsideFrustum(triangle) 
		&& !IsAllYOutsideFrustum(triangle) 
		&& !IsAllZOutsideFrustum(triangle));
}

GPU_CALLABLE static
bool IsVertexInFrustum(const FPoint4& NDC)
{
	return!((NDC.x < -1.f || NDC.x > 1.f) || 
			(NDC.y < -1.f || NDC.y > 1.f) || 
			(NDC.z < 0.f || NDC.z > 1.f));
}

GPU_CALLABLE static
bool IsTriangleInFrustum(const Triangle& triangle)
{
	return(IsVertexInFrustum(triangle.v0.p)
		|| IsVertexInFrustum(triangle.v1.p)
		|| IsVertexInFrustum(triangle.v2.p));
	//TODO: bug, triangles gets culled when zoomed in, aka all 3 vertices are outside of frustum
}

GPU_CALLABLE static
void NDCToScreenSpace(Triangle& triangle, const unsigned int width, const unsigned int height)
{
	triangle.v0.p.x = ((triangle.v0.p.x + 1) / 2) * width;
	triangle.v0.p.y = ((1 - triangle.v0.p.y) / 2) * height;
	triangle.v1.p.x = ((triangle.v1.p.x + 1) / 2) * width;
	triangle.v1.p.y = ((1 - triangle.v1.p.y) / 2) * height;
	triangle.v2.p.x = ((triangle.v2.p.x + 1) / 2) * width;
	triangle.v2.p.y = ((1 - triangle.v2.p.y) / 2) * height;
}

GPU_CALLABLE static
BoundingBox GetBoundingBox(const Triangle& triangle, const unsigned int width, const unsigned int height)
{
	BoundingBox bb;
	bb.xMin = (short)GetMinElement(triangle.v0.p.x, triangle.v1.p.x, triangle.v2.p.x) - 1; // xMin
	bb.yMin = (short)GetMinElement(triangle.v0.p.y, triangle.v1.p.y, triangle.v2.p.y) - 1; // yMin
	bb.xMax = (short)GetMaxElement(triangle.v0.p.x, triangle.v1.p.x, triangle.v2.p.x) + 1; // xMax
	bb.yMax = (short)GetMaxElement(triangle.v0.p.y, triangle.v1.p.y, triangle.v2.p.y) + 1; // yMax

	if (bb.xMin < 0) bb.xMin = 0; //clamp minX to Left of screen
	if (bb.yMin < 0) bb.yMin = 0; //clamp minY to Bottom of screen
	if (bb.xMax > width) bb.xMax = width; //clamp maxX to Right of screen
	if (bb.yMax > height) bb.yMax = height; //clamp maxY to Top of screen

	return bb;
}

GPU_CALLABLE GPU_INLINE static
RGBColor ShadePixel(const OVertex& oVertex, const GPUTextures& textures, SampleState sampleState, bool isDepthColour)
{
	return oVertex.c;

	RGBColor finalColour{};
	if (isDepthColour)
	{
		finalColour = RGBColor{ Remap(oVertex.p.z, 0.985f, 1.f), 0.f, 0.f }; // depth colour
		finalColour.ClampColor();
	}
	else
	{
		//TODO: textures lmao
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

#pragma endregion

#pragma region KERNELS
//Kernel launch params:	numBlocks, numThreadsPerBlock, numSharedMemoryBytes, stream

GPU_KERNEL
void ResetDepthBufferKernel(float dev_DepthBuffer[], const unsigned int width, const unsigned int height)
{
	//TODO: too many global accesses
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_DepthBuffer[pixelIdx] = 0.f; //FLT_MAX DEPTHBUFFER INVERTED INTERPRETATION
	}
}

GPU_KERNEL
void ClearFrameBufferKernel(unsigned int dev_FrameBuffer[], const unsigned int width, const unsigned int height, unsigned int colour)
{
	//TODO: too many global accesses
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_FrameBuffer[pixelIdx] = colour;
	}
}

GPU_KERNEL
void VertexShaderKernel(IVertex dev_IVertices[], OVertex dev_OVertices[], const size_t numVertices,
	const FPoint3 camPos, const FMatrix4 viewProjectionMatrix, const FMatrix4 worldMatrix)
{
	//TODO: local memory (global memory stats) is organised such that consecutive 32-bit variables are accessed by consecutive thread IDs
	//accesses are therefore fully coalesced as long as all threads in a warp access the same relative address
	//(e.g., same index in an array variable, same member in a structure variable)

	const unsigned int vertexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vertexIdx < numVertices)
	{
		const IVertex& iV = dev_IVertices[vertexIdx];
		//TODO: store in shared memory
		const OVertex oV = GetNDCVertex(iV, camPos, viewProjectionMatrix, worldMatrix);
		dev_OVertices[vertexIdx] = std::move(oV);
	}
	//TODO: coalesced global memory copy
}

GPU_KERNEL
void TriangleAssemblerKernel(Triangle dev_Triangles[], const OVertex const dev_OVertices[], const unsigned int const dev_IndexBuffer[], const size_t numIndices, 
	const PrimitiveTopology pt)
{
	//'talk about naming gore, eh?
	const unsigned int indexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	//triangles usually exist out of 3 vertices (sarcasm)
	if (pt == PrimitiveTopology::TriangleList)
	{
		if (indexIdx < numIndices / 3)
		{
			Triangle triangle;
			const unsigned int correctedIdx = indexIdx + 2;
			triangle.v0 = dev_OVertices[dev_IndexBuffer[correctedIdx - 2]];
			triangle.v1 = dev_OVertices[dev_IndexBuffer[correctedIdx - 1]];
			triangle.v2 = dev_OVertices[dev_IndexBuffer[correctedIdx]];
			dev_Triangles[indexIdx] = triangle;
		}
	}
	else //if (pt == PrimitiveTopology::TriangleStrip)
	{
		if (indexIdx < numIndices - 2)
		{
			Triangle triangle;
			const bool isOdd = indexIdx % 2 != 0;
			const unsigned int idx0{ dev_IndexBuffer[indexIdx] };
			const unsigned int idx1 = isOdd ? dev_IndexBuffer[indexIdx + 2] : dev_IndexBuffer[indexIdx + 1];
			const unsigned int idx2 = isOdd ? dev_IndexBuffer[indexIdx + 1] : dev_IndexBuffer[indexIdx + 2];
			triangle.v0 = dev_OVertices[dev_IndexBuffer[idx0]];
			triangle.v1 = dev_OVertices[dev_IndexBuffer[idx1]];
			triangle.v2 = dev_OVertices[dev_IndexBuffer[idx2]];
			dev_Triangles[indexIdx] = triangle;
		}
	}
}

GPU_KERNEL
void RasterizerKernel(const Triangle const dev_Triangles[], const size_t numTriangles, unsigned int dev_FrameBuffer[], float dev_DepthBuffer[], int dev_Mutex[],
	GPUTextures textures, SampleState sampleState, bool isDepthColour, const unsigned int width, const unsigned int height)
{
	//TODO: use shared memory, then coalesced copy
	//e.g. single bin buffer in single shared memory
	extern GPU_SHARED_MEMORY Triangle triangles[];
	//TODO: use binning, each bin their AABBs (and checks) (bin rasterizer)

	const unsigned int triangleIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (triangleIndex < numTriangles)
	{
		triangles[threadIdx.x] = dev_Triangles[triangleIndex];
	}
	__syncthreads();
	//https://stackoverflow.com/questions/6563261/how-to-use-coalesced-memory-access

	//Every thread processes 1 single triangle for now
	if (triangleIndex < numTriangles)
	{
		Triangle triangle = triangles[threadIdx.x];
		//TODO: add early out in triangle assembler?
		//Or clip
		if (!IsTriangleVisible(triangle))
			return;

		NDCToScreenSpace(triangle, width, height);
		const BoundingBox bb = GetBoundingBox(triangle, width, height);
		//Rasterize Screenspace triangle
			 
		//TODO: 1 thread per triangle is bad for performance, use binning
		//Loop over all pixels in bounding box
		for (unsigned short y = bb.yMin; y < bb.yMax; ++y)
		{
			for (unsigned short x = bb.xMin; x < bb.xMax; ++x)
			{
				const FPoint2 pixel{ float(x), float(y) };
				float weights[3];
				if (IsPixelInTriangle(triangle, pixel, weights))
				{
					const size_t pixelIdx = x + y * width;
					const float zInterpolated = (weights[0] * triangle.v0.p.z) + (weights[1] * triangle.v1.p.z) + (weights[2] * triangle.v2.p.z);
					if (DepthTest(dev_DepthBuffer, dev_Mutex, pixelIdx, zInterpolated))
					{
						OVertex oVertex;

						const OVertex& v0 = triangle.v0;
						const OVertex& v1 = triangle.v1;
						const OVertex& v2 = triangle.v2;

						const float wInterpolated = (weights[0] * v0.p.w) + (weights[1] * v1.p.w) + (weights[2] * v2.p.w);

						new (&oVertex.p) FPoint4{ pixel, zInterpolated, wInterpolated };

						new (&oVertex.uv) FVector2{
							weights[0] * (v0.uv.x / v0.p.w) + weights[1] * (v1.uv.x / v1.p.w) + weights[2] * (v2.uv.x / v2.p.w),
							weights[0] * (v0.uv.y / v0.p.w) + weights[1] * (v1.uv.y / v1.p.w) + weights[2] * (v2.uv.y / v2.p.w) };
						oVertex.uv *= wInterpolated;

						new (&oVertex.n) FVector3{
								weights[0] * (v0.n.x / v0.p.w) + weights[1] * (v1.n.x / v1.p.w) + weights[2] * (v2.n.x / v2.p.w),
								weights[0] * (v0.n.y / v0.p.w) + weights[1] * (v1.n.y / v1.p.w) + weights[2] * (v2.n.y / v2.p.w),
								weights[0] * (v0.n.z / v0.p.w) + weights[1] * (v1.n.z / v1.p.w) + weights[2] * (v2.n.z / v2.p.w) };
						oVertex.n *= wInterpolated;

						new (&oVertex.tan) const FVector3{
							weights[0] * (v0.tan.x / v0.p.w) + weights[1] * (v1.tan.x / v1.p.w) + weights[2] * (v2.tan.x / v2.p.w),
							weights[0] * (v0.tan.y / v0.p.w) + weights[1] * (v1.tan.y / v1.p.w) + weights[2] * (v2.tan.y / v2.p.w),
							weights[0] * (v0.tan.z / v0.p.w) + weights[1] * (v1.tan.z / v1.p.w) + weights[2] * (v2.tan.z / v2.p.w) };

						new (&oVertex.vd) FVector3{
						weights[0] * (v0.vd.y / v0.p.w) + weights[1] * (v1.vd.y / v1.p.w) + weights[2] * (v2.vd.y / v2.p.w),
						weights[0] * (v0.vd.x / v0.p.w) + weights[1] * (v1.vd.x / v1.p.w) + weights[2] * (v2.vd.x / v2.p.w),
						weights[0] * (v0.vd.z / v0.p.w) + weights[1] * (v1.vd.z / v1.p.w) + weights[2] * (v2.vd.z / v2.p.w) };
						Normalize(oVertex.vd);

						//new (&oVertex.c) const RGBColor{ v0.c * weights[0] + v1.c * weights[1] + v2.c * weights[2] };

						new (&oVertex.c) const RGBColor{
							weights[0] * v0.c.r + weights[1] * v1.c.r + weights[2] * v2.c.r,
							weights[0] * v0.c.g + weights[1] * v1.c.g + weights[2] * v2.c.g,
							weights[0] * v0.c.b + weights[1] * v1.c.b + weights[2] * v2.c.b };

						//Pixel Shading
						//const RGBColor colour = ShadePixel(oVertex, textures, sampleState, isDepthColour);
						const RGBA rgba{ oVertex.c };
						dev_FrameBuffer[pixelIdx] = rgba.colour;
					}
				}
			}
		}
	}
}

GPU_KERNEL
void PixelShaderKernel(unsigned int dev_FrameBuffer[], GPUTextures textures, SampleState sampleState, 
	bool isDepthColour, const unsigned int width, const unsigned int height)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		//const OVertex& oVertex = dev_PixelShaderBuffer[pixelIdx]; //copy or ref?
		//const RGBColor colour = ShadePixel(oVertex, textures, sampleState, isDepthColour);
		//store individual bytes from 32-bit format colour (RGBA)
		//convert to 4-byte RGBA value
		RGBA rgba{ { 1.f, 1.f, 1.f } };
		dev_FrameBuffer[pixelIdx] = rgba.colour;
	}
}

#pragma endregion

#pragma region KERNEL LAUNCHERS

CPU_CALLABLE
void CUDARenderer::Clear(const RGBColor& colour)
{
	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	ResetDepthBufferKernel<<<numBlocks, numThreadsPerBlock>>>
		(dev_DepthBuffer, m_WindowHelper.Width, m_WindowHelper.Height);
	const RGBA rgba{ colour };
	ClearFrameBufferKernel<<<numBlocks, numThreadsPerBlock>>>
		(dev_FrameBuffer, m_WindowHelper.Width, m_WindowHelper.Height, rgba.colour);
	//cudaMemsetAsync(dev_FrameBuffer, rgba.colour, m_WindowHelper.Width * m_WindowHelper.Height * sizeof(unsigned int));
}

CPU_CALLABLE
void CUDARenderer::VertexShader(const MeshIdentifier& mi, const FPoint3& camPos, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix)
{
	const size_t numVertices = mi.pMesh->GetVertices().size();
	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocksForVertices = (numVertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
	VertexShaderKernel<<<numBlocksForVertices, numThreadsPerBlock>>>(
		dev_IVertexBuffer[mi.Idx], dev_OVertexBuffer[mi.Idx], numVertices,
		camPos, viewProjectionMatrix, worldMatrix);
}

CPU_CALLABLE
void CUDARenderer::TriangleAssembler(const MeshIdentifier& mi)
{
	const size_t numIndices = mi.pMesh->GetIndexes().size();
	const PrimitiveTopology topology = mi.pMesh->GetTopology();
	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = ((unsigned int)numIndices + numThreadsPerBlock - 1) / numThreadsPerBlock;
	TriangleAssemblerKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_Triangles, dev_OVertexBuffer[mi.Idx], dev_IndexBuffer[mi.Idx], numIndices, topology);
}

CPU_CALLABLE
void CUDARenderer::Rasterizer(GPUTextures& textures, SampleState sampleState, bool isDepthColour)
{
	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = ((unsigned int)m_NumTriangles - 1) / numThreadsPerBlock + 1;
	const size_t numSharedMemoryBytesPerBlock = (sizeof(Triangle) * m_NumTriangles) / numBlocks;
	RasterizerKernel<<<numBlocks, numThreadsPerBlock, numSharedMemoryBytesPerBlock >>>(
		dev_Triangles, m_NumTriangles,
		dev_FrameBuffer, dev_DepthBuffer, dev_Mutex,
		textures, sampleState, isDepthColour,
		m_WindowHelper.Width, m_WindowHelper.Height);
}

CPU_CALLABLE
void CUDARenderer::PixelShader(GPUTextures& textures, SampleState sampleState, bool isDepthColour)
{
	//TODO: if ever want to reuse again, get OVertex buffer
	const dim3 numThreadsPerBlock{ 16, 16 };
	const dim3 numBlocks{ m_WindowHelper.Width / numThreadsPerBlock.x, m_WindowHelper.Height / numThreadsPerBlock.y };
	PixelShaderKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_FrameBuffer, textures,
		sampleState, isDepthColour,
		m_WindowHelper.Width, m_WindowHelper.Height);
}

#pragma endregion

#pragma region PUBLIC FUNCTIONS

CPU_CALLABLE
void CUDARenderer::LoadScene(const SceneGraph* pSceneGraph)
{
	m_NumTriangles = 0;
	FreeMeshBuffers(); //!must be called before MeshIdentifiers.clear()!
	const std::vector<Mesh*>& pMeshes = pSceneGraph->GetObjects();
	for (const Mesh* pMesh : pMeshes)
	{
		MeshIdentifier mi{};
		mi.Idx = m_MeshIdentifiers.size();
		mi.pMesh = pMesh;
		m_MeshIdentifiers.push_back(mi);

		const std::vector<IVertex>& vertexBuffer = pMesh->GetVertices();
		const std::vector<unsigned int>& indexBuffer = pMesh->GetIndexes();
		const size_t numVertices = vertexBuffer.size();
		const size_t numIndices = indexBuffer.size();
		const PrimitiveTopology topology = pMesh->GetTopology();
		const FMatrix4& worldMat = pMesh->GetWorldMatrix();

		AllocateMeshBuffers(numVertices, numIndices, mi.Idx);
		CopyMeshBuffers(vertexBuffer, indexBuffer, mi.Idx);

		switch (topology)
		{
		case PrimitiveTopology::TriangleList:
			m_NumTriangles += numIndices / 3;
			break;
		case PrimitiveTopology::TriangleStrip:
			m_NumTriangles += numIndices - 2;
			break;
		default:
			break;
		}
	}

	CheckErrorCuda(cudaFree(dev_Triangles)); //Free unwanted memory
	//Allocate device memory for entire range of triangles
	CheckErrorCuda(cudaMalloc((void**)&dev_Triangles, m_NumTriangles * sizeof(Triangle)));
}

CPU_CALLABLE
void CUDARenderer::Render(const SceneManager& sm, const Camera* pCamera)
{
#ifdef _DEBUG
	if (EnterValidRenderingState())
		exit(1);
#else
	EnterValidRenderingState();
#endif

	//Render Data
	const bool isDepthColour = sm.IsDepthColour();
	const SampleState sampleState = sm.GetSampleState();

	//Camera Data
	const FPoint3& camPos = pCamera->GetPos();
	const FMatrix4 lookatMatrix = pCamera->GetLookAtMatrix();
	const FMatrix4 viewMatrix = pCamera->GetViewMatrix(lookatMatrix);
	const FMatrix4 projectionMatrix = pCamera->GetProjectionMatrix();
	const FMatrix4 viewProjectionMatrix = projectionMatrix * viewMatrix;

	//TODO: use renderdata as constant memory

	//TODO: random illegal memory access BUG
	//Update global memory for camera's matrices
	//UpdateCameraDataAsync(camPos, viewProjectionMatrix);

	//SceneGraph Data
	const SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pObjects = pSceneGraph->GetObjects();

	//TODO: create big coalesced memory array of buffer(s)?
	for (const MeshIdentifier& mi : m_MeshIdentifiers)
	{
		//Mesh Data
		const Mesh* pMesh = pObjects[mi.Idx];
		const FMatrix4& worldMat = pMesh->GetWorldMatrix();

		//TODO: random illegal memory access BUG
		//Update global memory for mesh's worldmatrix
		//UpdateWorldMatrixDataAsync(worldMat);
		//cudaDeviceSynchronize();

		//StartTimer();

		//---STAGE 1---:  Perform Output Vertex Assembling
		//TODO: async & streams
		//TODO: find out what order is best, for cudaDevCpy and Malloc
		VertexShader(mi, camPos, viewProjectionMatrix, worldMat);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 1---

		//std::cout << "Vertex Shading total time: " << StopTimer() << "ms\n";
		//StartTimer();

		//---STAGE 2---:  Perform Triangle Assembling
		TriangleAssembler(mi);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 2---

		//std::cout << "Triangle Assembling total time: " << StopTimer() << "ms\n";
		//StartTimer();

		//---STAGE 3---: Peform Triangle Rasterization & Pixel Shading
		const Textures& textures = pMesh->GetTextures();
		GPUTextures gpuTextures{};
		Rasterizer(gpuTextures, sampleState, isDepthColour);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 3---

		//std::cout << "Rasterizer total time: " << StopTimer() << "ms\n";
	}

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

CPU_CALLABLE
void CUDARenderer::WarmUp()
{
	ResetDepthBufferKernel<<<0, 0>>>(nullptr, m_WindowHelper.Width, m_WindowHelper.Height);
	ClearFrameBufferKernel<<<0, 0>>>(nullptr, m_WindowHelper.Width, m_WindowHelper.Height, 0);
	VertexShaderKernel<<<0, 0>>>(nullptr, nullptr, 0, {}, {}, {});
	TriangleAssemblerKernel<<<0, 0>>>(nullptr, nullptr, nullptr, 0, (PrimitiveTopology)0);
	RasterizerKernel<<<0, 0>>>(nullptr, 0, nullptr, nullptr, nullptr,
		{}, (SampleState)0, false, m_WindowHelper.Width, m_WindowHelper.Height);
}

#pragma endregion