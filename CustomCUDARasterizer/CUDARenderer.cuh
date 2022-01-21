#pragma once
//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"
#include <vector>

#include "Math.h"
#include "RGBColor.h"

struct WindowHelper;
class Camera;
class Mesh;
struct IVertex;
struct OVertex;
struct BoundingBox;
enum class SampleState;
struct GPUTextures;
class SceneManager;

//////////////////////////////
//-----RAII Wrapper Class-----
//////////////////////////////

class CUDARenderer final
{
public:
	CPU_CALLABLE CUDARenderer(const WindowHelper& windowHelper);
	CPU_CALLABLE ~CUDARenderer() noexcept;

	CPU_CALLABLE CUDARenderer(const CUDARenderer&) = delete;
	CPU_CALLABLE CUDARenderer(CUDARenderer&&) noexcept = delete;
	CPU_CALLABLE CUDARenderer& operator=(const CUDARenderer&) = delete;
	CPU_CALLABLE CUDARenderer& operator=(CUDARenderer&&) noexcept = delete;

	//function that launches the kernels
	CPU_CALLABLE void Render(const SceneManager& sm, const Camera* pCamera);

	struct RenderData
	{
		FPoint3 camPos;
		FMatrix4 projectionMatrix;
		FMatrix4 viewMatrix;
		FMatrix4 worldMatrix;
	};

private:
	//-----MEMBER VARIABLES-----

	const WindowHelper& m_WindowHelper;
	IVertex* m_dev_IVertexBuffer{};
	unsigned int* m_dev_IndexBuffer{};
	OVertex* m_dev_OVertexBuffer{};
	OVertex* m_dev_PixelShaderBuffer{};
	RGBColor* m_dev_FrameBuffer{};
	float* m_dev_DepthBuffer{};
	float* m_dev_RenderData;
	dim3 m_NumThreadsPerBlock{ 16, 16 };
	dim3 m_NumBlocks{};

	//-----CPU HELPER FUNCTIONS-----
	
	//function that allocates input buffers from mesh
	CPU_CALLABLE void AllocateMeshBuffers(const size_t numVertices, const size_t numIndices);
	//function that copies output buffers vertex shader
	CPU_CALLABLE void CopyMeshBuffers(const std::vector<IVertex>& vertexBuffer, const std::vector<unsigned int>& indexBuffer);
	//function that frees mesh buffers
	CPU_CALLABLE void FreeMeshBuffers();
	//function that allocates buffers
	CPU_CALLABLE void InitCUDARasterizer();
	//function that frees all allocated buffers
	CPU_CALLABLE void FreeCUDARasterizer();

	//function that updates camera's matrices
	CPU_CALLABLE void UpdateCameraData(const FPoint3& camPos, const FMatrix4& projectionMatrix, const FMatrix4& viewMatrix);
	//function that updates mesh's worldmatrix
	CPU_CALLABLE void UpdateWorldMatrixData(const FMatrix4& worldMatrix);

	//Swap out buffers
	CPU_CALLABLE int SwapBuffers();
	//Present new frame
	CPU_CALLABLE int Present();

	//-----KERNEL LAUNCHERS-----

	//Reset depth buffer and clear framebuffer
	CPU_CALLABLE void Clear(const RGBColor& colour = { 0.25f, 0.25f, 0.25f });
	CPU_CALLABLE void VertexShader(const Mesh* pMesh, const FPoint3& camPos, const FMatrix4& projectionMatrix, const FMatrix4& viewMatrix);
	CPU_CALLABLE void Rasterizer(const size_t numVertices, const size_t numIndices);
	CPU_CALLABLE void PixelShader(const GPUTextures& textures, SampleState sampleState, bool isDepthColour);
};