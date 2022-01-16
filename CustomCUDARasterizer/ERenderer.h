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

#include "Vertex.h"
#include "Mesh.h"

struct SDL_Window;
struct SDL_Surface;

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

	private:
		SDL_Window* m_pWindow = nullptr;
		SDL_Surface* m_pFrontBuffer = nullptr;
		SDL_Surface* m_pBackBuffer = nullptr;
		uint32_t* m_pBackBufferPixels = nullptr;
		uint32_t m_Width = 0;
		uint32_t m_Height = 0;
		float m_RasterScreenSpaceX[3]{}, m_RasterScreenSpaceY[3]{};
		float* m_pDepthBuffer{};

		void CreateDepthBuffer();
		void ClearDepthBuffer();
		void ClearScreen();
		void BlackDraw(uint32_t c, uint32_t r);

		bool FrustumTest(const OVertex NDC[3]);
		bool FrustumTestVertex(const OVertex& NDC);
		void SetVerticesToRasterScreenSpace(const OVertex triangle[3]);
		void RenderPixelsInTriangle(const OVertex triangle[3], const Mesh::Textures& textures, const uint32_t boundingValues[4]);
		void RenderPixelsInTriangle(const OVertex triangle[3], const Mesh::Textures& textures);
		bool IsPixelInTriangle(const FPoint2& pixel, float weights[3]);
		bool DepthTest(const OVertex triangle[3], float& depthBuffer, float weights[3], float& zInterpolated);
		void SetBoundingBox(uint32_t boundingValues[4]);
		OVertex GetNDCVertex(const IVertex& vertex, const FMatrix4& worldMatrix = FMatrix4::Identity());
		std::vector<OVertex> GetNDCMeshVertices(const std::vector<IVertex>& vertices, const FMatrix4& worldMatrix = FMatrix4::Identity());
		void ShadePixel(const OVertex& oVertex, const Mesh::Textures& textures);
	};
}

#endif