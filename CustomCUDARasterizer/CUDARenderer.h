#pragma once
#include <vector>
#include <functional>

#include "Math.h"
#include "RGBColor.h"
#include "CullingMode.h"
#include "CUDABenchMarker.h"
#include "CUDAAtomicQueue.cuh"
#include "CUDAWindowHelper.h"

struct WindowHelper;
class Camera;
class Mesh;
struct IVertex;
struct OVertex;
struct BoundingBox;
enum class SampleState;
class SceneManager;
class SceneGraph;
class CUDATextureManager;
struct CUDATexturesCompact;
class CUDAMesh;
struct PixelShade;

//////////////////////////////
//-----RAII Wrapper Class-----
//////////////////////////////

class CUDARenderer final
{
public:
	CUDARenderer(const WindowHelper& windowHelper);
	~CUDARenderer() noexcept;

	CUDARenderer(const CUDARenderer&) = delete;
	CUDARenderer(CUDARenderer&&) noexcept = delete;
	CUDARenderer& operator=(const CUDARenderer&) = delete;
	CUDARenderer& operator=(CUDARenderer&&) noexcept = delete;

	//Testing purposes
	void DrawTexture(char* tp);
	//Testing purposes
	void DrawTextureGlobal(char* tp, bool isStretchedToWindow = true, SampleState sampleState = (SampleState)0);

	//Preload and store scene in persistent memory
	//This will eliminate overhead by loading mesh data and accessing global memory
	void LoadScene(const SceneGraph* pSceneGraph, const CUDATextureManager& tm);

	//Inits the binner object which handles its internal binning buffers on the GPU
	void SetupBins(const IPoint2& numBins, const IPoint2& binDim, unsigned int binQueueMaxSize);

	//Lock backbuffer surface and call Clear
	int EnterValidRenderingState();
	//function that launches the kernels and outputs to buffers
	void Render(const SceneManager& sm, const CUDATextureManager& tm, const Camera* pCamera);
	//Update window screen
	void Present();
	//function that launches the kernels and directly outputs to window
	void RenderAuto(const SceneManager& sm, const CUDATextureManager& tm, const Camera* pCamera);

	unsigned int GetTotalNumVisibleTriangles() const { return m_TotalVisibleNumTriangles; }
	unsigned int GetTotalNumTriangles() const { return m_TotalNumTriangles; }

	//function that outputs GPU specs
	void DisplayGPUSpecs(int deviceId = 0);

	//function that launches all kernels to eliminate overhead time (used for measuring)
	void WarmUp();

	void StartTimer();
	float StopTimer();
	CUDABenchMarker& GetBenchMarker() { return m_BenchMarker; }

private:
	//-----MEMBER VARIABLES-----

	unsigned int m_TotalNumTriangles{};
	unsigned int m_TotalVisibleNumTriangles{};
	const WindowHelper& m_WindowHelper;
	unsigned int* m_Dev_NumVisibleTriangles{};
	IPoint2 m_BinDim;
	CUDABenchMarker m_BenchMarker{};
	CUDAAtomicQueues<unsigned int> m_BinQueues;
	CUDAWindowHelper<PixelShade> m_CUDAWindowHelper;
	std::vector<CUDAMesh*> m_pCUDAMeshes{};
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

	//-----CPU HELPER FUNCTIONS-----
	
	//function that allocates device buffers
	void AllocateCUDADeviceBuffers();
	//function that allocates and copies host mesh buffers to device
	CUDAMesh* AllocateCUDAMeshBuffers(const Mesh* pMesh);
	//function that frees all device mesh buffers
	void FreeAllCUDAMeshBuffers();
	//function that frees a device mesh buffers
	void FreeCUDAMeshBuffers(CUDAMesh* pCudaMesh);
	//function that frees device buffers
	void FreeCUDADeviceBuffers();

	//function that updates camera's matrices
	void UpdateCameraDataAsync(const FPoint3& camPos, const FVector3& camFwd);
	//function that updates mesh's worldmatrix
	void UpdateWorldMatrixDataAsync(const FMatrix4& worldMatrix, const FMatrix4& wvpMat, const FMatrix3& rotationMat);
	//function that pre-stores a mesh's textures
	void UpdateCUDAMeshTextures(CUDAMesh* pCudaMesh, const CUDATextureManager& tm);
	//function that fetches a mesh's textures
	CUDATexturesCompact GetCUDAMeshTextures(const int* texIds, const CUDATextureManager& tm);

	//function that blocks host calls until stream has finished
	void WaitForStream(cudaStream_t stream);
	//function that checks whether stream has finished without blocking host
	bool IsStreamFinished(cudaStream_t stream);

	//-----KERNEL LAUNCHERS-----

	//Reset depth buffer, mutex buffer and pixelshadebuffer
	void Clear(const RGBColor& colour = { 0.25f, 0.25f, 0.25f });
	void VertexShader(const CUDAMesh* pCudaMesh);
	void TriangleAssembler(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm = CullingMode::BackFace, cudaStream_t stream = cudaStreamDefault);
	void TriangleBinner(const CUDAMesh* pCudaMesh, const unsigned int triangleIdxOffset = 0, cudaStream_t stream = cudaStreamDefault);
	void TriangleAssemblerAndBinner(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm = CullingMode::BackFace, cudaStream_t stream = cudaStreamDefault);
	void Rasterizer(const CUDAMesh* pCudaMesh, const FVector3& camFwd, const CullingMode cm = CullingMode::BackFace, cudaStream_t stream = cudaStreamDefault);
	void PixelShader(SampleState sampleState, bool isDepthColour);
};