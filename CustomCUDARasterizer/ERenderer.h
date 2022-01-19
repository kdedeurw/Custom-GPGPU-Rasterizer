/*=============================================================================*/
// Copyright 2017-2019 Elite Engine
// Authors: Matthieu Delaere
/*=============================================================================*/
// ERenderer.h: class that holds the surface to render to, does traverse the pixels 
// and traces the rays using a tracer
/*=============================================================================*/
#ifndef ELITE_RAYTRACING_RENDERER
#define	ELITE_RAYTRACING_RENDERER

#include <cstdint>
#include <vector>

#include "Mesh.h"

struct SDL_Window;
struct SDL_Surface;

class Camera;
struct BoundingBox;
struct IVertex;
struct OVertex;
namespace Elite
{
	class Renderer final
	{
	public:
		Renderer(SDL_Window* pWindow);
		~Renderer();

		Renderer(const Renderer&) = delete;
		Renderer(Renderer&&) noexcept = delete;
		Renderer& operator=(const Renderer&) = delete;
		Renderer& operator=(Renderer&&) noexcept = delete;

		void Render();
		bool SaveBackbufferToImage() const;
		void SetCamera(Camera* pCamera) { m_pCamera = pCamera; };

	private:
		SDL_Window* m_pWindow = nullptr;
		SDL_Surface* m_pFrontBuffer = nullptr;
		SDL_Surface* m_pBackBuffer = nullptr;
		uint32_t* m_pBackBufferPixels = nullptr;
		uint32_t m_Width = 0;
		uint32_t m_Height = 0;
		float* m_pDepthBuffer{};
		Camera* m_pCamera{};
		const Mesh::Textures* m_pTextures{};

		void CreateDepthBuffer();
		inline void ClearDepthBuffer();
		inline void ClearScreen();
		inline void BlackDraw(unsigned short c, unsigned short r);

		inline void RenderTriangle(OVertex* NDCTriangle[3]);

		bool FrustumTest(OVertex* NDC[3]);
		inline bool FrustumTestVertex(const OVertex& NDC);
		inline void SetVerticesToRasterScreenSpace(OVertex* triangle[3]);
		inline void RenderPixelsInTriangle(OVertex* screenspaceTriangle[3]);
		inline bool IsPixelInTriangle(OVertex* screenspaceTriangle[3], const FPoint2& pixel, float weights[3]);
		inline bool DepthTest(OVertex* triangle[3], float& depthBuffer, float weights[3], float& zInterpolated);
		BoundingBox GetBoundingBox(OVertex* screenspaceTriangle[3]);
		OVertex GetNDCVertexDeprecated(const IVertex& vertex, const FMatrix4& worldMatrix = FMatrix4::Identity());
		inline OVertex GetNDCVertex(const IVertex& vertex, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix = FMatrix4::Identity());
		std::vector<OVertex> GetNDCMeshVertices(const std::vector<IVertex>& vertices, const FMatrix4& iewProjectionMatrix, const FMatrix4& worldMatrix = FMatrix4::Identity());
		inline void ShadePixel(const OVertex& oVertex, const Mesh::Textures& textures);
	};
}

#endif