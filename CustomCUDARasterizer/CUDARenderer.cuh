#pragma once
//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"
#include <vector>

#include "Math.h"
#include "RGBColor.h"
#include "PrimitiveTopology.h"
#include "GPUTextures.h"

struct WindowHelper;
class Camera;
class Mesh;
struct IVertex;
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

	//Testing purposes
	void DrawTexture(char* tp);
	//Testing purposes
	void DrawTextureGlobal(char* tp, bool isStretchedToWindow = true);

	//Preload and store scene in persistent memory
	//This will eliminate overhead by loading mesh data and accessing global memory
	CPU_CALLABLE void LoadScene(const SceneGraph* pSceneGraph);

	//Lock backbuffer surface and call Clear
	CPU_CALLABLE int EnterValidRenderingState();
	//function that launches the kernels and outputs to buffers
	CPU_CALLABLE void Render(const SceneManager& sm, const Camera* pCamera);
	//Update window screen
	CPU_CALLABLE void Present();
	//function that launches the kernels and directly outputs to window
	CPU_CALLABLE void RenderAuto(const SceneManager& sm, const Camera* pCamera);

	//function that launches all kernels to eliminate overhead time (used for measuring)
	CPU_CALLABLE void WarmUp();

	//sets initial timepoint to measure between
	CPU_CALLABLE void StartTimer();
	//returns time in between start and stop in ms
	CPU_CALLABLE float StopTimer();

	struct MeshIdentifier
	{
		size_t Idx; //<READ ONLY>
		size_t NumTriangles; //<READ ONLY>
		const Mesh* pMesh; //<READ ONLY>
		GPUTextures Textures; //<READ ONLY>
	};
private:
	//-----MEMBER VARIABLES-----

	const WindowHelper& m_WindowHelper;
	size_t m_TotalNumTriangles{};
	float m_TimerMs{};
	unsigned int* m_h_pFrameBuffer{};
	cudaEvent_t m_StartEvent{}, m_StopEvent{};
	std::vector<MeshIdentifier> m_MeshIdentifiers{};
	std::vector<GPUTextures> m_TextureObjects{};

	//CANNOT DIRECTLY COPY PINNED MEMORY TO CONST DEVICE MEMORY
	//CameraData* m_pCameraData{};
	//FMatrix4* m_pWorldMatrixData{};

	//-----CPU HELPER FUNCTIONS-----
	
	//function that allocates all output buffers for a mesh (idx)
	CPU_CALLABLE void AllocateMeshBuffers(const size_t numVertices, const size_t numIndices, const size_t numTriangles, size_t meshIdx = 0);
	//function that copies raw input buffers for a mesh (idx)
	CPU_CALLABLE void CopyMeshBuffers(const std::vector<IVertex>& vertexBuffer, const std::vector<unsigned int>& indexBuffer, size_t meshIdx = 0);
	//function that preloads GPU textures in device's texture memory
	CPU_CALLABLE void LoadMeshTextures(const std::string texturePaths[4], size_t meshIdx = 0);
	//function that frees all texture objects
	CPU_CALLABLE void FreeTextures();
	//function that frees all mesh buffers
	CPU_CALLABLE void FreeMeshBuffers();
	//function that allocates device buffers
	CPU_CALLABLE void InitCUDADeviceBuffers();
	//function that frees device buffers
	CPU_CALLABLE void FreeCUDADeviceBuffers();
	//function that outputs GPU specs
	CPU_CALLABLE void DisplayGPUSpecs(int deviceId = 0);

	//function that updates camera's matrices
	CPU_CALLABLE void UpdateCameraDataAsync(const FPoint3& camPos, const FMatrix4& viewProjectionMatrix);
	//function that updates mesh's worldmatrix
	CPU_CALLABLE void UpdateWorldMatrixDataAsync(const FMatrix4& worldMatrix);

	//-----KERNEL LAUNCHERS-----

	//Reset depth buffer, mutex buffer and pixelshadebuffer
	CPU_CALLABLE void Clear(const RGBColor& colour = { 0.25f, 0.25f, 0.25f });
	CPU_CALLABLE void VertexShader(const MeshIdentifier& mi, const FPoint3& camPos, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix);
	CPU_CALLABLE void TriangleAssembler(const MeshIdentifier& mi);
	CPU_CALLABLE void Rasterizer(const MeshIdentifier& mi);
	CPU_CALLABLE void PixelShader(bool isDepthColour);
};