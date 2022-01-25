#pragma once
//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"
#include <vector>

#include "Math.h"
#include "RGBColor.h"
#include "PrimitiveTopology.h"

struct WindowHelper;
class Camera;
class Mesh;
struct IVertex;
struct OVertex;
struct BoundingBox;
enum class SampleState;
struct GPUTextures;
class SceneManager;
class SceneGraph;

//////////////////////////////
//-----RAII Wrapper Class-----
//////////////////////////////

class CUDARenderer final
{
	//-----STRUCT DECLARATIONS-----

	struct CameraData
	{
		FPoint3 camPos;
		FMatrix4 viewProjectionMatrix;
	};
	union CameraDataRaw
	{
		float* data;
		CameraData cameraData;
	};

public:
	CPU_CALLABLE CUDARenderer(const WindowHelper& windowHelper);
	CPU_CALLABLE ~CUDARenderer() noexcept;

	CPU_CALLABLE CUDARenderer(const CUDARenderer&) = delete;
	CPU_CALLABLE CUDARenderer(CUDARenderer&&) noexcept = delete;
	CPU_CALLABLE CUDARenderer& operator=(const CUDARenderer&) = delete;
	CPU_CALLABLE CUDARenderer& operator=(CUDARenderer&&) noexcept = delete;

	//Preload and store scene in persistent memory
	//This will eliminate overhead by loading mesh data and accessing global memory
	CPU_CALLABLE void LoadScene(const SceneGraph* pSceneGraph);

	//function that launches the kernels
	CPU_CALLABLE void Render(const SceneManager& sm, const Camera* pCamera);

	//function that launches all kernels to eliminate overhead time (used for measuring)
	CPU_CALLABLE void WarmUp();

	//sets initial timepoint to measure between
	CPU_CALLABLE void StartTimer();
	//returns time in between start and stop in ms
	CPU_CALLABLE float StopTimer();

	struct MeshIdentifier
	{
		size_t Idx;
		const Mesh* pMesh;
	};
private:
	//-----MEMBER VARIABLES-----

	const WindowHelper& m_WindowHelper;
	size_t m_NumTriangles{};
	float m_TimerMs{};
	unsigned int* m_h_pFrameBuffer{};
	cudaEvent_t m_StartEvent{}, m_StopEvent{};
	std::vector<MeshIdentifier> m_MeshIdentifiers{};

	//CANNOT DIRECTLY COPY PINNED MEMORY TO CONST DEVICE MEMORY
	//CameraData* m_pCameraData{};
	//FMatrix4* m_pWorldMatrixData{};

	//-----CPU HELPER FUNCTIONS-----
	
	//function that allocates input buffers from mesh
	CPU_CALLABLE void AllocateMeshBuffers(const size_t numVertices, const size_t numIndices, size_t meshIdx = 0);
	//function that copies output buffers vertex shader
	CPU_CALLABLE void CopyMeshBuffers(const std::vector<IVertex>& vertexBuffer, const std::vector<unsigned int>& indexBuffer, size_t meshIdx = 0);
	//function that frees mesh buffers
	CPU_CALLABLE void FreeMeshBuffers();
	//function that allocates buffers
	CPU_CALLABLE void InitCUDARasterizer();
	//function that frees all allocated buffers
	CPU_CALLABLE void FreeCUDARasterizer();
	//function that outputs GPU specs
	CPU_CALLABLE void DisplayGPUSpecs(int deviceId = 0);

	//function that updates camera's matrices
	CPU_CALLABLE void UpdateCameraDataAsync(const FPoint3& camPos, const FMatrix4& viewProjectionMatrix);
	//function that updates mesh's worldmatrix
	CPU_CALLABLE void UpdateWorldMatrixDataAsync(const FMatrix4& worldMatrix);

	//Lock backbuffer surface and call Clear
	CPU_CALLABLE int EnterValidRenderingState();
	//Update window screen
	CPU_CALLABLE void Present();

	//-----KERNEL LAUNCHERS-----

	//Reset depth buffer and clear framebuffer
	CPU_CALLABLE void Clear(const RGBColor& colour = { 0.25f, 0.25f, 0.25f });
	CPU_CALLABLE void VertexShader(const MeshIdentifier& mi, const FPoint3& camPos, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix);
	CPU_CALLABLE void TriangleAssembler(const MeshIdentifier& mi);
	CPU_CALLABLE void Rasterizer();
	//DEPRECATED
	CPU_CALLABLE void PixelShader(GPUTextures& textures, SampleState sampleState, bool isDepthColour);
};

//TODO: replace Clear with CPU calls cudamemset()

//TODO: mutex buffer and atomicAdd() for depthbuffer

//TODO: host pinned memory without SDL window pixelbuffer
//SDL allows random access to pixelbuffer, but cuda does not allowed host memory to be there