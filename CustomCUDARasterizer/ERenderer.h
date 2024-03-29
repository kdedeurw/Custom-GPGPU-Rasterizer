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

struct SDL_Window;
struct SDL_Surface;

class Camera;
struct BoundingBox;
struct IVertex;
struct OVertex;
class SceneManager;
struct Textures;
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

		void Render(const SceneManager& sm);
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
		const Textures* m_pTextures{};

		void CreateDepthBuffer();
		inline void ClearDepthBuffer();
		inline void ClearScreen();
		inline void BlackDraw(unsigned short c, unsigned short r);

		inline void RenderTriangle(const SceneManager& sm, const OVertex* triangle[3], FPoint4 rasterCoords[3]);

		bool IsAllXOutsideFrustum(FPoint4 NDC[3]) const;
		bool IsAllYOutsideFrustum(FPoint4 NDC[3]) const;
		bool IsAllZOutsideFrustum(FPoint4 NDC[3]) const;
		bool IsTriangleVisible(FPoint4 NDC[3]) const;
		bool IsTriangleInFrustum(FPoint4 NDCs[3]) const;
		inline bool IsVertexInFrustum(const FPoint4& NDC) const;
		inline void NDCToScreenSpace(FPoint4& rasterCoords);
		inline void RenderPixelsInTriangle(const SceneManager& sm, const OVertex* triangle[3], FPoint4 rasterCoords[3]);
		inline bool IsPixelInTriangle(FPoint4 rasterCoords[3], const FPoint2& pixel, float weights[3]);
		inline bool DepthTest(FPoint4 rasterCoords[3], float& depthBuffer, float weights[3], float& zInterpolated);
		BoundingBox GetBoundingBox(FPoint4 rasterCoords[3]);
		OVertex GetNDCVertexDeprecated(const IVertex& vertex, const FMatrix4& worldMatrix = FMatrix4::Identity());
		inline OVertex GetNDCVertex(const IVertex& vertex, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix = FMatrix4::Identity());
		std::vector<OVertex> GetNDCMeshVertices(const std::vector<IVertex>& vertices, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix = FMatrix4::Identity());
		std::vector<OVertex> GetNDCMeshVertices(const float* pVertices, unsigned int numVertices, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix = FMatrix4::Identity());
		inline void ShadePixel(const OVertex& oVertex, const Textures& textures, const SceneManager& sm);
	};
}

#endif