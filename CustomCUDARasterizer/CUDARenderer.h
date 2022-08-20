#pragma once
#include <vector>

#include "Math.h"
#include "RGBColor.h"
#include "GPUTextures.h"
#include "CullingMode.h"
#include "CUDABenchMarker.h"
#include "CUDAAtomicQueue.cuh"

struct WindowHelper;
class Camera;
class Mesh;
struct IVertex;
struct IVertex_Point4;
struct OVertex;
struct BoundingBox;
enum class SampleState;
class SceneManager;
class SceneGraph;

//////////////////////////////
//-----RAII Wrapper Class-----
//////////////////////////////

class CUDARenderer final
{
public:
	CUDARenderer(const WindowHelper& windowHelper, IPoint2 numBins = {}, IPoint2 binDim = {}, unsigned int binQueueMaxSize = 0);
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
	void LoadScene(const SceneGraph* pSceneGraph);

	//Lock backbuffer surface and call Clear
	int EnterValidRenderingState();
	//function that launches the kernels and outputs to buffers
	void Render(const SceneManager& sm, const Camera* pCamera);
	//Update window screen
	void Present();
	//function that launches the kernels and directly outputs to window
	void RenderAuto(const SceneManager& sm, const Camera* pCamera);

	unsigned int GetTotalNumVisibleTriangles() const { return m_TotalVisibleNumTriangles; }
	unsigned int GetTotalNumTriangles() const { return m_TotalNumTriangles; }

	//function that outputs GPU specs
	void DisplayGPUSpecs(int deviceId = 0);

	//function that launches all kernels to eliminate overhead time (used for measuring)
	void WarmUp();

	void StartTimer();
	float StopTimer();
	CUDABenchMarker& GetBenchMarker() { return m_BenchMarker; }

	struct MeshIdentifier
	{
		unsigned int Idx; //<READ ONLY>
		unsigned int TotalNumTriangles; //<READ ONLY>
		unsigned int VisibleNumTriangles; //<READ/WRITE>
		const Mesh* pMesh; //<READ ONLY>
		GPUTexturesCompact Textures; //<READ ONLY>
	};
private:
	//-----MEMBER VARIABLES-----

	const WindowHelper& m_WindowHelper;
	unsigned int m_TotalNumTriangles{};
	unsigned int m_TotalVisibleNumTriangles{};
	unsigned int* m_h_pFrameBuffer{};
	IPoint2 m_BinDim;
	CUDABenchMarker m_BenchMarker{};
	CUDAAtomicQueues<unsigned int> m_BinQueues;
	std::vector<MeshIdentifier> m_MeshIdentifiers{};
	std::vector<GPUTexturesCompact> m_TextureObjects{};

	//-----CPU HELPER FUNCTIONS-----
	
	//function that allocates all output buffers for a mesh (idx)
	void AllocateMeshBuffers(const size_t numVertices, const size_t numIndices, const size_t numTriangles, unsigned int stride, size_t meshIdx = 0);
	//function that copies raw input buffers for a mesh (idx)
	void CopyMeshBuffers(const float* vertexBuffer, unsigned int numVertices, short stride, const unsigned int* indexBuffer, unsigned int numIndices, size_t meshIdx = 0);
	//function that preloads GPU textures in device's texture memory
	GPUTexturesCompact LoadMeshTextures(const std::string texturePaths[4], size_t meshIdx = 0);
	//function that loads GPU textures in device's texture memory
	GPUTexture LoadGPUTexture(const std::string texturePath, unsigned int textureIdx);
	//function that frees all texture objects
	void FreeTextures();
	//function that frees all mesh buffers
	void FreeMeshBuffers();
	//function that allocates device buffers
	void InitCUDADeviceBuffers();
	//function that frees device buffers
	void FreeCUDADeviceBuffers();

	//function that updates camera's matrices
	void UpdateCameraDataAsync(const FPoint3& camPos, const FVector3& camFwd);
	//function that updates mesh's worldmatrix
	void UpdateWorldMatrixDataAsync(const FMatrix4& worldMatrix, const FMatrix4& wvpMat, const FMatrix3& rotationMat);

	//-----KERNEL LAUNCHERS-----

	//Reset depth buffer, mutex buffer and pixelshadebuffer
	void Clear(const RGBColor& colour = { 0.25f, 0.25f, 0.25f });
	void VertexShader(const MeshIdentifier& mi);
	void TriangleAssembler(MeshIdentifier& mi, const FVector3& camFwd, const CullingMode cm = CullingMode::BackFace, cudaStream_t stream = cudaStreamDefault);
	void TriangleBinner(MeshIdentifier& mi, cudaStream_t stream = cudaStreamDefault);
	void TriangleAssemblerAndBinner(MeshIdentifier& mi, const FVector3& camFwd, const CullingMode cm = CullingMode::BackFace, cudaStream_t stream = cudaStreamDefault);
	void Rasterizer(const MeshIdentifier& mi, const FVector3& camFwd, const CullingMode cm = CullingMode::BackFace, cudaStream_t stream = cudaStreamDefault);
	void PixelShader(SampleState sampleState, bool isDepthColour);
};