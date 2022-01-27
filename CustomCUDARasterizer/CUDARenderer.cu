#include "PCH.h"
#include "CUDARenderer.cuh"
#include <vector>

#include "DEFINES.h"

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
#include "Light.h"
#include "GPUTextures.h"
#include "PrimitiveTopology.h"
#include "CullingMode.h"

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

struct RasterTriangle
{
	FPoint4 v0;
	FPoint4 v1;
	FPoint4 v2;
};

struct TriangleIdx
{
	unsigned int idx0;
	unsigned int idx1;
	unsigned int idx2;
	bool isCulled;
	//padded
};

struct PixelShade
{
	unsigned int colour;
	float zInterpolated;
	float wInterpolated;
	FVector2 uv;
	FVector3 n;
	FVector3 tan;
	FVector3 vd;
	GPUTextures textures;
};

#pragma endregion

#pragma region GLOBAL VARIABLES

constexpr int NumTextures = 4;

//CONST DEVICE MEMORY - Does NOT have to be allocated or freed
GPU_CONST_MEMORY static
float dev_RenderData_const[sizeof(RenderData) / sizeof(float)]{};
//NOTE: cannot contain anything else besides primitive variables (int, float, etc.)

//DEVICE MEMORY - Does have to be allocated and freed
static unsigned int* dev_FrameBuffer{};
static int* dev_DepthBuffer{}; //defined as INTEGER type for atomicCAS to work properly
static int* dev_Mutex{};
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
void CUDARenderer::AllocateMeshBuffers(const size_t numVertices, const size_t numIndices, const size_t numTriangles, size_t meshIdx)
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
	CheckErrorCuda(cudaMalloc((void**)&dev_IVertexBuffer[meshIdx], numVertices * sizeof(IVertex)));
	//Allocate Index Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_IndexBuffer[meshIdx], numIndices * sizeof(unsigned int)));
	//Allocate Ouput Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&dev_OVertexBuffer[meshIdx], numVertices * sizeof(OVertex)));
	//Allocate device memory for entire range of triangles
	CheckErrorCuda(cudaMalloc((void**)&dev_Triangles[meshIdx], numTriangles * sizeof(TriangleIdx)));
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

	GPUTextures gpuTextures{};

	//0 DIFFUSE > 1 NORMAL > 2 SPECULAR > 3 GLOSSINESS
	for (int i{}; i < NumTextures; ++i)
	{
		GPUTexture* gpuTexture;
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
		SDL_Surface* pSurface = IMG_Load(texturePaths[i].c_str());
		if (pSurface)
		{
			const unsigned int width = pSurface->w;
			const unsigned int height = pSurface->h;
			const unsigned int* pixels = (unsigned int*)pSurface->pixels;
			const int bpp = pSurface->format->BytesPerPixel;
			//const size_t sizeInBytes = width * height * bpp;
			const size_t textureIdx = meshIdx * NumTextures + i;

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

			gpuTexture->dev_pTex = dev_TextureObject;
			gpuTexture->w = width;
			gpuTexture->h = height;
			gpuTexture->dev_TextureData = dev_TextureData[textureIdx];

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
	}
	//store textures
	m_TextureObjects[meshIdx] = gpuTextures;
}

CPU_CALLABLE
void CUDARenderer::FreeTextures()
{
	//destroy all texture objects
	for (const GPUTextures& textures : m_TextureObjects)
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

	cudaFree(dev_Mutex);
	dev_Mutex = nullptr;

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
float EdgeFunction(const FPoint2& v, const FVector2& edge, const FPoint2& pixel)
{
	// clockwise
	const FVector2 vertexToPixel{ pixel - v };
	return Cross(vertexToPixel, edge);
}

GPU_CALLABLE static
bool IsPixelInTriangle(const RasterTriangle& triangle, const FPoint2& pixel, float weights[3])
{
	const FPoint2& v0 = triangle.v0.xy;
	const FPoint2& v1 = triangle.v1.xy;
	const FPoint2& v2 = triangle.v2.xy;

	const FVector2 edgeA{ v1 - v0 };
	const FVector2 edgeB{ v2 - v1 };
	const FVector2 edgeC{ v0 - v2 };
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

	weights[2] = EdgeFunction(v0, edgeA, pixel);
	weights[0] = EdgeFunction(v1, edgeB, pixel);
	weights[1] = EdgeFunction(v2, edgeC, pixel);

	return weights[0] >= 0.f && weights[1] >= 0.f && weights[2] >= 0.f;
}

GPU_CALLABLE static
//UNUSED
void ModifyPixelShadeBuffer()
{

}

GPU_CALLABLE static
//DEPRECATED
//This will "block" the current thread into a while loop until depthtest is complete
bool IsDepthTestSucceeded(float dev_DepthBuffer[], int dev_Mutex[], const size_t pixelIdx, float zInterpolated)
{
	//TODO: shared memory
	
	//Depth Test with correct depth interpolation
	if (zInterpolated < 0 || zInterpolated > 1.f)
		return false;
	
	//Update depthbuffer atomically
	bool isDepthTestSucceeded = false;
	bool isDone = false;
	do
	{
		isDone = (atomicCAS(&dev_Mutex[pixelIdx], 0, 1) == 0);
		if (isDone)
		{
			//critical section
			if (zInterpolated > dev_DepthBuffer[pixelIdx]) //DEPTH BUFFER INVERTED INTERPRETATION
			{
				dev_DepthBuffer[pixelIdx] = zInterpolated;
				//TODO: (atomically) write to pixelshaderbuffer
				isDepthTestSucceeded = true;
			}
			dev_Mutex[pixelIdx] = 0;
			//end of critical section
		}
	} while (!isDone);
	return isDepthTestSucceeded;
	//atomicCAS
	/*
	//int atomicCAS(int* address, int compare, int val);
	//reads the 16 - bit, 32 - bit or 64 - bit word old located at the address address in global or shared memory, 
	//computes(old == compare ? val : old), and stores the result back to memory at the same address.
	//These three operations are performed in one atomic transaction.The function returns old(Compare And Swap).
	*/
}

GPU_CALLABLE static
bool IsAllXOutsideFrustum(const RasterTriangle& triangle)
{
	return	(triangle.v0.x < -1.f && triangle.v1.x < -1.f && triangle.v2.x < -1.f) ||
			(triangle.v0.x > 1.f && triangle.v1.x > 1.f && triangle.v2.x > 1.f);
}

GPU_CALLABLE static
bool IsAllYOutsideFrustum(const RasterTriangle& triangle)
{
	return	(triangle.v0.y < -1.f && triangle.v1.y < -1.f && triangle.v2.y < -1.f) ||
			(triangle.v0.y > 1.f && triangle.v1.y > 1.f && triangle.v2.y > 1.f);
}

GPU_CALLABLE static
bool IsAllZOutsideFrustum(const RasterTriangle& triangle)
{
	return	(triangle.v0.z < 0.f && triangle.v1.z < 0.f && triangle.v2.z < 0.f) ||
			(triangle.v0.z > 1.f && triangle.v1.z > 1.f && triangle.v2.z > 1.f);
}

GPU_CALLABLE static
bool IsTriangleVisible(const RasterTriangle& triangle)
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
bool IsTriangleInFrustum(const RasterTriangle& triangle)
{
	return(IsVertexInFrustum(triangle.v0)
		|| IsVertexInFrustum(triangle.v1)
		|| IsVertexInFrustum(triangle.v2));
	//TODO: bug, triangles gets culled when zoomed in, aka all 3 vertices are outside of frustum
}

GPU_CALLABLE static
void NDCToScreenSpace(RasterTriangle& triangle, const unsigned int width, const unsigned int height)
{
	triangle.v0.x = ((triangle.v0.x + 1) / 2) * width;
	triangle.v0.y = ((1 - triangle.v0.y) / 2) * height;
	triangle.v1.x = ((triangle.v1.x + 1) / 2) * width;
	triangle.v1.y = ((1 - triangle.v1.y) / 2) * height;
	triangle.v2.x = ((triangle.v2.x + 1) / 2) * width;
	triangle.v2.y = ((1 - triangle.v2.y) / 2) * height;
}

GPU_CALLABLE static
BoundingBox GetBoundingBox(const RasterTriangle& triangle, const unsigned int width, const unsigned int height)
{
	BoundingBox bb;
	bb.xMin = (short)GetMinElement(triangle.v0.x, triangle.v1.x, triangle.v2.x) - 1; // xMin
	bb.yMin = (short)GetMinElement(triangle.v0.y, triangle.v1.y, triangle.v2.y) - 1; // yMin
	bb.xMax = (short)GetMaxElement(triangle.v0.x, triangle.v1.x, triangle.v2.x) + 1; // xMax
	bb.yMax = (short)GetMaxElement(triangle.v0.y, triangle.v1.y, triangle.v2.y) + 1; // yMax

	if (bb.xMin < 0) bb.xMin = 0; //clamp minX to Left of screen
	if (bb.yMin < 0) bb.yMin = 0; //clamp minY to Bottom of screen
	if (bb.xMax > width) bb.xMax = width; //clamp maxX to Right of screen
	if (bb.yMax > height) bb.yMax = height; //clamp maxY to Top of screen

	return bb;
}

GPU_CALLABLE GPU_INLINE static
RGBColor ShadePixel(const PixelShade& pixelShade, SampleState sampleState, bool isDepthColour)
{
	RGBColor finalColour{};
	if (isDepthColour)
	{
		//TODO: replace with depthbuffer kernel call?
		finalColour = RGBColor{ Remap(pixelShade.zInterpolated, 0.985f, 1.f), 0.f, 0.f }; // depth colour
	}
	else
	{
		//global settings
		bool isFlipGreenChannel = false;
		const RGBColor ambientColour{ 0.05f, 0.05f, 0.05f };
		const FVector3 lightDirection = { 0.577f, -0.577f, -0.577f };
		const float lightIntensity = 7.0f;

		// texture sampling
		const GPUTexture& diffTex = pixelShade.textures.Diff;
		const GPUTexture& normTex = pixelShade.textures.Norm;
		const GPUTexture& specTex = pixelShade.textures.Spec;
		const GPUTexture& glossTex = pixelShade.textures.Gloss;
		if (diffTex.dev_pTex != 0)
		{
			const RGBColor diffuseSample = GPUTextureSampler::Sample(diffTex, pixelShade.uv, sampleState);

			if (normTex.dev_pTex != 0)
			{
				const RGBColor normalSample = GPUTextureSampler::Sample(normTex, pixelShade.uv, sampleState);

				// normal mapping
				FVector3 binormal = Cross(pixelShade.tan, pixelShade.n);
				if (isFlipGreenChannel)
					binormal = -binormal;
				const FMatrix3 tangentSpaceAxis{ pixelShade.tan, binormal, pixelShade.n };

				FVector3 finalNormal{ 2.f * normalSample.r - 1.f, 2.f * normalSample.g - 1.f, 2.f * normalSample.b - 1.f };
				finalNormal = tangentSpaceAxis * finalNormal;

				// light calculations
				float observedArea{ Dot(-finalNormal, lightDirection) };
				Clamp(observedArea, 0.f, observedArea);
				observedArea /= (float)PI;
				observedArea *= lightIntensity;
				const RGBColor diffuseColour = diffuseSample * observedArea;

				if (specTex.dev_pTex != 0 && glossTex.dev_pTex != 0)
				{
					const RGBColor specularSample = GPUTextureSampler::Sample(specTex, pixelShade.uv, sampleState);
					const RGBColor glossSample = GPUTextureSampler::Sample(glossTex, pixelShade.uv, sampleState);

					// phong specular
					const FVector3 reflectV{ Reflect(lightDirection, finalNormal) };
					float angle{ Dot(reflectV, pixelShade.vd) };
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
		}
		else
		{
			finalColour = GetRGBColor_SDL(pixelShade.colour);
		}
	}
	return finalColour;
}

#pragma endregion

#pragma region KERNELS
//Kernel launch params:	numBlocks, numThreadsPerBlock, numSharedMemoryBytes, stream

GPU_KERNEL
void ResetDepthBufferKernel(int* dev_DepthBuffer, const unsigned int width, const unsigned int height)
{
	//TODO: too many global accesses
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		dev_DepthBuffer[pixelIdx] = INT_MAX;
	}
}

GPU_KERNEL
void ClearFrameBufferKernel(unsigned int* dev_FrameBuffer, const unsigned int width, const unsigned int height, unsigned int colour)
{
	//TODO: too many global accesses
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		memcpy(&dev_FrameBuffer[pixelIdx], &colour, sizeof(unsigned int));
	}
}

GPU_KERNEL
void ClearPixelShadeBufferKernel(PixelShade* dev_PixelShadeBuffer, const unsigned int width, const unsigned int height)
{
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		memset(&dev_PixelShadeBuffer[pixelIdx], 0, sizeof(PixelShade));
	}
}

GPU_KERNEL
void VertexShaderKernel(const IVertex* __restrict__ dev_IVertices, OVertex* dev_OVertices, const size_t numVertices,
	const FPoint3 camPos, const FMatrix4 viewProjectionMatrix, const FMatrix4 worldMatrix)
{
	//TODO: local memory (global memory stats) is organised such that consecutive 32-bit variables are accessed by consecutive thread IDs
	//accesses are therefore fully coalesced as long as all threads in a warp access the same relative address
	//(e.g., same index in an array variable, same member in a structure variable)

	const unsigned int vertexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vertexIdx < numVertices)
	{
		const IVertex& iV = dev_IVertices[vertexIdx];
		const OVertex oV = GetNDCVertex(iV, camPos, viewProjectionMatrix, worldMatrix);
		dev_OVertices[vertexIdx] = std::move(oV);
	}
}

GPU_KERNEL
void TriangleAssemblerKernel(TriangleIdx* dev_Triangles, const unsigned int* __restrict__ const dev_IndexBuffer, const size_t numIndices, 
	OVertex* dev_OVertices, const PrimitiveTopology pt)
{
	//'talk about naming gore, eh?
	const unsigned int indexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pt == PrimitiveTopology::TriangleList)
	{
		const unsigned int correctedIdx = (indexIdx * 3);
		if (correctedIdx < numIndices)
		{
			TriangleIdx triangle;
			triangle.idx0 = dev_IndexBuffer[correctedIdx];
			triangle.idx1 = dev_IndexBuffer[correctedIdx + 1];
			triangle.idx2 = dev_IndexBuffer[correctedIdx + 2];
			triangle.isCulled = false;
			dev_Triangles[indexIdx] = triangle;
			//atomically increment visible triangle count
			//atomicAdd(dev_VisibleNumTriangles, 1);
		}
	}
	else //if (pt == PrimitiveTopology::TriangleStrip)
	{
		if (indexIdx < numIndices - 2)
		{
			TriangleIdx triangle;
			const bool isOdd = (indexIdx % 2);
			triangle.idx0 = dev_IndexBuffer[indexIdx];
			triangle.idx1 = isOdd ? dev_IndexBuffer[indexIdx + 2] : dev_IndexBuffer[indexIdx + 1];
			triangle.idx2 = isOdd ? dev_IndexBuffer[indexIdx + 1] : dev_IndexBuffer[indexIdx + 2];
			triangle.isCulled = false;
			dev_Triangles[indexIdx] = triangle;
			//atomically increment visible triangle count
			//atomicAdd(dev_VisibleNumTriangles, 1);
		}
	}
}

GPU_KERNEL
void RasterizerKernel(const TriangleIdx* __restrict__ const dev_Triangles, const OVertex* __restrict__ const dev_OVertices, const size_t numTriangles,
	PixelShade* dev_PixelShadeBuffer, int* dev_DepthBuffer, int* dev_Mutex, GPUTextures textures, 
	const FVector3 camFwd, const CullingMode cm, const unsigned int width, const unsigned int height)
{
	//TODO: use shared memory, then coalesced copy
	//e.g. single bin buffer in single shared memory
	//extern GPU_SHARED_MEMORY Triangle triangles[];
	//TODO: use binning, each bin their AABBs (and checks) (bin rasterizer)

	const unsigned int triangleIndex = threadIdx.x + blockIdx.x * blockDim.x;
	//if (triangleIndex < numTriangles)
	//{
	//	triangles[threadIdx.x] = dev_Triangles[triangleIndex];
	//}
	//__syncthreads();
	//https://stackoverflow.com/questions/6563261/how-to-use-coalesced-memory-access

	//Every thread processes 1 single triangle for now
	if (triangleIndex < numTriangles)
	{
		TriangleIdx triangleIdx = dev_Triangles[triangleIndex];

		//if (triangleIdx.isCulled)
		//	return;

		const OVertex v0 = dev_OVertices[triangleIdx.idx0];
		const OVertex v1 = dev_OVertices[triangleIdx.idx1];
		const OVertex v2 = dev_OVertices[triangleIdx.idx2];

		//is triangle visible according to cullingmode?
		if (cm == CullingMode::BackFace)
		{
			const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
			const float cullingValue = Dot(camFwd, faceNormal);
			//is back facing triangle?
			if (cullingValue <= 0.f)
			{
				return; //cull triangle
			}
		}
		else if (cm == CullingMode::FrontFace)
		{
			//TODO: render entire object but invert all triangles
			//const OVertex& v0Inv = NDCVertices[indices[idx]];
			//const OVertex& v1Inv = NDCVertices[indices[idx - 1]];
			//const OVertex& v2Inv = NDCVertices[indices[idx - 2]];

			const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
			const float cullingValue = Dot(camFwd, faceNormal);
			////is front facing triangle?
			if (cullingValue >= 0.f)
			{
				return; // cull triangle
			}
		}
		//else //if (cm == CullingMode::NoCulling)
		//{}

		RasterTriangle rasterTriangle;
		rasterTriangle.v0 = v0.p;
		rasterTriangle.v1 = v1.p;
		rasterTriangle.v2 = v2.p;

		if (!IsTriangleVisible(rasterTriangle))
		{
			return;
		}

		NDCToScreenSpace(rasterTriangle, width, height);
		const BoundingBox bb = GetBoundingBox(rasterTriangle, width, height);
		//Rasterize Screenspace triangle

		const float v0InvDepth = 1.f / rasterTriangle.v0.w;
		const float v1InvDepth = 1.f / rasterTriangle.v1.w;
		const float v2InvDepth = 1.f / rasterTriangle.v2.w;
		
		//TODO: 1 thread per triangle is bad for performance, use binning
		//Loop over all pixels in bounding box
		for (unsigned short y = bb.yMin; y < bb.yMax; ++y)
		{
			for (unsigned short x = bb.xMin; x < bb.xMax; ++x)
			{
				const FPoint2 pixel{ float(x), float(y) };
				float weights[3];
				if (IsPixelInTriangle(rasterTriangle, pixel, weights))
				{
					const float totalArea = abs(Cross(rasterTriangle.v0.xy - rasterTriangle.v1.xy, rasterTriangle.v0.xy - rasterTriangle.v2.xy));
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
					new (&pixelShade.uv) FVector2{
						weights[0] * (v0.uv.x * v0InvDepth) + weights[1] * (v1.uv.x * v1InvDepth) + weights[2] * (v2.uv.x * v2InvDepth),
						weights[0] * (v0.uv.y * v0InvDepth) + weights[1] * (v1.uv.y * v1InvDepth) + weights[2] * (v2.uv.y * v2InvDepth) };
					pixelShade.uv *= wInterpolated;

					//normal
					new (&pixelShade.n) FVector3{
							weights[0] * (v0.n.x * v0InvDepth) + weights[1] * (v1.n.x * v1InvDepth) + weights[2] * (v2.n.x * v2InvDepth),
							weights[0] * (v0.n.y * v0InvDepth) + weights[1] * (v1.n.y * v1InvDepth) + weights[2] * (v2.n.y * v2InvDepth),
							weights[0] * (v0.n.z * v0InvDepth) + weights[1] * (v1.n.z * v1InvDepth) + weights[2] * (v2.n.z * v2InvDepth) };
					pixelShade.n *= wInterpolated;

					//tangent
					new (&pixelShade.tan) FVector3{
						weights[0] * (v0.tan.x * v0InvDepth) + weights[1] * (v1.tan.x * v1InvDepth) + weights[2] * (v2.tan.x * v2InvDepth),
						weights[0] * (v0.tan.y * v0InvDepth) + weights[1] * (v1.tan.y * v1InvDepth) + weights[2] * (v2.tan.y * v2InvDepth),
						weights[0] * (v0.tan.z * v0InvDepth) + weights[1] * (v1.tan.z * v1InvDepth) + weights[2] * (v2.tan.z * v2InvDepth) };

					//view direction
					new (&pixelShade.vd) FVector3{
						weights[0] * (v0.vd.x * v0InvDepth) + weights[1] * (v1.vd.x * v1InvDepth) + weights[2] * (v2.vd.x * v2InvDepth),
						weights[0] * (v0.vd.y * v0InvDepth) + weights[1] * (v1.vd.y * v1InvDepth) + weights[2] * (v2.vd.y * v2InvDepth),
						weights[0] * (v0.vd.z * v0InvDepth) + weights[1] * (v1.vd.z * v1InvDepth) + weights[2] * (v2.vd.z * v2InvDepth) };
					Normalize(pixelShade.vd);

					//colour
					const RGBColor interpolatedColour{
						weights[0] * v0.c.r + weights[1] * v1.c.r + weights[2] * v2.c.r,
						weights[0] * v0.c.g + weights[1] * v1.c.g + weights[2] * v2.c.g,
						weights[0] * v0.c.b + weights[1] * v1.c.b + weights[2] * v2.c.b };
					pixelShade.colour = GetRGBA_SDL(interpolatedColour).colour;

					//store textures
					pixelShade.textures = textures;

					//multiplying z value by a INT_MAX because atomicCAS only accepts ints
					const int scaledZ = zInterpolated * INT_MAX;

					//Perform atomic depth test
					bool isDone = false;
					do
					{
						isDone = (atomicCAS(&dev_Mutex[pixelIdx], 0, 1) == 0);
						if (isDone)
						{
							//critical section
							if (scaledZ < dev_DepthBuffer[pixelIdx])
							{
								//update depthbuffer
								dev_DepthBuffer[pixelIdx] = scaledZ;
								//modify pixelshadebuffer
								dev_PixelShadeBuffer[pixelIdx] = pixelShade;
							}
							dev_Mutex[pixelIdx] = 0;
							//end of critical section
						}
					} while (!isDone);
				}
			}
		}
	}
}

GPU_KERNEL
void PixelShaderKernel(unsigned int* dev_FrameBuffer, const PixelShade* __restrict__ const dev_PixelShadeBuffer,
	SampleState sampleState, bool isDepthColour, const unsigned int width, const unsigned int height)
{
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < width && y < height)
	{
		const unsigned int pixelIdx = x + y * width;
		PixelShade pixelShade = dev_PixelShadeBuffer[pixelIdx];
		const RGBColor colour = ShadePixel(pixelShade, sampleState, isDepthColour);
		RGBA rgba{ colour };
		dev_FrameBuffer[pixelIdx] = rgba.colour;
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
		unsigned char b = rgba.values.b;
		rgba.values.b = rgba.values.r;
		rgba.values.r = b;
		dev_FrameBuffer[pixelIdx] = rgba.colour;
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
	const RGBA rgba{ colour };

	ResetDepthBufferKernel<<<numBlocks, numThreadsPerBlock>>>
		(dev_DepthBuffer, m_WindowHelper.Width, m_WindowHelper.Height);
	//ClearFrameBufferKernel<<<numBlocks, numThreadsPerBlock>>>
	//	(dev_FrameBuffer, m_WindowHelper.Width, m_WindowHelper.Height, rgba.colour);
	//ClearPixelShadeBufferKernel<<<numBlocks, numThreadsPerBlock>>>
	//	(dev_PixelShadeBuffer, m_WindowHelper.Width, m_WindowHelper.Height);

	CheckErrorCuda(cudaMemset(dev_Mutex, 0, m_WindowHelper.Width * m_WindowHelper.Height * sizeof(int)));
	CheckErrorCuda(cudaMemset(dev_PixelShadeBuffer, 0, m_WindowHelper.Width * m_WindowHelper.Height * sizeof(PixelShade)));
}

CPU_CALLABLE
void CUDARenderer::VertexShader(const MeshIdentifier& mi, const FPoint3& camPos, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix)
{
	const size_t numVertices = mi.pMesh->GetVertices().size();
	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = (unsigned int)(numVertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
	VertexShaderKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_IVertexBuffer[mi.Idx], dev_OVertexBuffer[mi.Idx], numVertices,
		camPos, viewProjectionMatrix, worldMatrix);
}

CPU_CALLABLE
void CUDARenderer::TriangleAssembler(MeshIdentifier& mi)
{
	//unsigned int* dev_VisibleNumTriangles;
	//CheckErrorCuda(cudaMalloc((void**)&dev_VisibleNumTriangles, sizeof(unsigned int)));

	const size_t numIndices = mi.pMesh->GetIndexes().size();
	const PrimitiveTopology topology = mi.pMesh->GetTopology();

	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = ((unsigned int)numIndices + numThreadsPerBlock - 1) / numThreadsPerBlock;
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
	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocks = ((unsigned int)m_TotalNumTriangles - 1) / numThreadsPerBlock + 1;
	//const size_t numSharedMemoryBytesPerBlock = (sizeof(TriangleIdx) * m_TotalNumTriangles) / numBlocks;
	RasterizerKernel<<<numBlocks, numThreadsPerBlock>>>(
		dev_Triangles[mi.Idx], dev_OVertexBuffer[mi.Idx], mi.TotalNumTriangles,
		dev_PixelShadeBuffer, dev_DepthBuffer, dev_Mutex, mi.Textures,
		camFwd, cm, m_WindowHelper.Width, m_WindowHelper.Height);
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

#pragma endregion

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
	const std::vector<Mesh*>& pMeshes = pSceneGraph->GetObjects();
	for (const Mesh* pMesh : pMeshes)
	{
		MeshIdentifier mi{};
		mi.Idx = m_MeshIdentifiers.size();
		mi.pMesh = pMesh;
		size_t numTriangles{};

		const std::vector<IVertex>& vertexBuffer = pMesh->GetVertices();
		const std::vector<unsigned int>& indexBuffer = pMesh->GetIndexes();
		const size_t numVertices = vertexBuffer.size();
		const size_t numIndices = indexBuffer.size();
		const PrimitiveTopology topology = pMesh->GetTopology();
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
		CopyMeshBuffers(vertexBuffer, indexBuffer, mi.Idx);
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

	//SceneGraph Data
	const SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pObjects = pSceneGraph->GetObjects();

	m_TotalVisibleNumTriangles = 0;
	for (MeshIdentifier& mi : m_MeshIdentifiers)
	{
		//Mesh Data
		const Mesh* pMesh = pObjects[mi.Idx];
		const FMatrix4& worldMat = pMesh->GetWorldMatrix();

		//TODO: random illegal memory access BUG
		//Update global memory for mesh's worldmatrix
		//UpdateWorldMatrixDataAsync(worldMat);
		//cudaDeviceSynchronize();

#ifdef BENCHMARK
		StartTimer();
#endif
		//---STAGE 1---:  Perform Output Vertex Assembling
		//TODO: async & streams
		//TODO: find out what order is best, for cudaDevCpy and Malloc
		VertexShader(mi, camPos, viewProjectionMatrix, worldMat);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 1---
#ifdef BENCHMARK
		float vertexShadingMs = StopTimer();
		StartTimer();
#endif
		//---STAGE 2---:  Perform Triangle Assembling
		TriangleAssembler(mi);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 2---
#ifdef BENCHMARK
		float TriangleAssemblingMs = StopTimer();
		StartTimer();
#endif
		//---STAGE 3---: Peform Triangle Rasterization & Pixel Shading
		Rasterizer(mi, camFwd, cm);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 3---
#ifdef BENCHMARK
		float RasterizationMs = StopTimer();
		StartTimer();
#endif
		//---STAGE 3---: Peform  Pixel Shading
		PixelShader(sampleState, isDepthColour);
		CheckErrorCuda(cudaDeviceSynchronize());
		//---END STAGE 3---
#ifdef BENCHMARK
		float PixelShadingMs = StopTimer();
		std::cout << "VS: " << vertexShadingMs << "ms | TA: " << TriangleAssemblingMs << "ms | Raster: " << RasterizationMs << "ms | PS: " << PixelShadingMs << "ms\r";
#endif
	}
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

CPU_CALLABLE
void CUDARenderer::WarmUp()
{
	ResetDepthBufferKernel<<<0, 0>>>(nullptr, m_WindowHelper.Width, m_WindowHelper.Height);
	ClearFrameBufferKernel<<<0, 0>>>(nullptr, m_WindowHelper.Width, m_WindowHelper.Height, 0);
	VertexShaderKernel<<<0, 0>>>(nullptr, nullptr, 0, {}, {}, {});
	TriangleAssemblerKernel<<<0, 0>>>(nullptr, nullptr, 0, nullptr, (PrimitiveTopology)0);
	RasterizerKernel<<<0, 0>>>(nullptr, nullptr, 0, nullptr, nullptr, nullptr, {}, {}, (CullingMode)0, m_WindowHelper.Width, m_WindowHelper.Height);
	PixelShaderKernel<<<0, 0>>> (nullptr, nullptr, SampleState(0), false, m_WindowHelper.Width, m_WindowHelper.Height);
}

#pragma endregion