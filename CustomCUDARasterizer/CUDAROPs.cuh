#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Math.h"
#include "MathUtilities.h"
#include "RGBColor.h"
#include "Mesh.h"

#include "GPUHelpers.h"

class IVertex;
class OVertex;
class Camera;
struct BoundingBox;
struct SDL_Window;
struct SDL_Surface;
enum class SampleState;
class GPUTexture;

namespace CUDAROP
{
	struct GPUTextures
	{
		GPUTexture* pDiff{};
		GPUTexture* pNorm{};
		GPUTexture* pSpec{};
		GPUTexture* pGloss{};
	};

	//Helpers
	GPU_CALLABLE bool FrustumTest(const OVertex* NDC[3]);
	GPU_CALLABLE bool FrustumTestVertex(const OVertex& NDC);
	GPU_CALLABLE void NDCToScreenSpace(FPoint4 rasterCoords[3], const unsigned int width, const unsigned int height);
	GPU_CALLABLE bool EdgeFunction(const FVector2& v0, const FVector2& v1, const FPoint2& pixel);
	GPU_CALLABLE bool IsPixelInTriangle(FPoint4 rasterCoords[3], const FPoint2& pixel, float weights[3]);
	GPU_CALLABLE bool DepthTest(FPoint4 rasterCoords[3], float& depthBuffer, float weights[3], float& zInterpolated);

	GPU_CALLABLE OVertex GetNDCVertex(const IVertex& vertex, const FPoint3& camPos, const FMatrix4& viewMatrix, const FMatrix4& projectionMatrix, const FMatrix4& worldMatrix = FMatrix4::Identity());
	GPU_CALLABLE OVertex* GetNDCMeshVertices(const IVertex vertices[], const size_t size, const IVertex& vertex, const FPoint3& camPos, const FMatrix4& viewMatrix, const FMatrix4& projectionMatrix, const FMatrix4& worldMatrix = FMatrix4::Identity());
	GPU_CALLABLE BoundingBox GetBoundingBox(FPoint4 rasterCoords[3], const unsigned int width, const unsigned int height);

	//Rendering
	GPU_CALLABLE void RenderPixelsInTriangle(OVertex* screenspaceTriangle[3], FPoint4 rasterCoords[3], SDL_Surface* pBuffer, float* pDepthBuffer, const BoundingBox& bb, const unsigned int width, const unsigned int height);
	GPU_CALLABLE RGBColor ShadePixel(const OVertex& oVertex, const GPUTextures& textures, SampleState sampleState, bool isDepthColour);
	GPU_CALLABLE void RasterizeTriangle(OVertex* triangle[3], FPoint4 rasterCoords[3], SDL_Surface* pBuffer, float* pDepthBuffer, const unsigned int width, const unsigned int height);
}