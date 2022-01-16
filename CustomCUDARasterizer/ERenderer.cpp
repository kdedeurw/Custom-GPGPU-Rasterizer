#include "PCH.h"
//External includes
#include "SDL.h"
#include "SDL_surface.h"

//Project includes
#include "ERenderer.h"
#include "ERGBColor.h"
#include <iostream>
#include "SceneManager.h"
#include "SceneGraph.h"
#include "Camera.h"
#include "Mesh.h"
#include "Texture.h"
#include "ObjParser.h"
#include "DirectionalLight.h"
#include "EventManager.h"

Elite::Renderer::Renderer(SDL_Window * pWindow)
{
	//Initialize
	m_pWindow = pWindow;
	m_pFrontBuffer = SDL_GetWindowSurface(pWindow);
	int width, height = 0;
	SDL_GetWindowSize(pWindow, &width, &height);
	m_Width = static_cast<uint32_t>(width);
	m_Height = static_cast<uint32_t>(height);
	m_pBackBuffer = SDL_CreateRGBSurface(0, m_Width, m_Height, 32, 0, 0, 0, 0);
	m_pBackBufferPixels = (uint32_t*)m_pBackBuffer->pixels;
	//
	CreateDepthBuffer();
}

Elite::Renderer::~Renderer()
{
	if (m_pDepthBuffer)
		delete m_pDepthBuffer;
	m_pDepthBuffer = nullptr;
}

void Elite::Renderer::Render()
{
	SDL_LockSurface(m_pBackBuffer);
	ClearDepthBuffer();
	ClearScreen();

	//Loop over all objects
	for (Mesh* pMesh : SceneManager::GetInstance()->GetSceneGraph()->GetObjects())
	{
		std::vector<OVertex> NDCVertices{ std::move(GetNDCMeshVertices(pMesh->GetVertices(), pMesh->GetWorldMatrix())) }; // calculate all IVertices to OVertices !in NDC!
		const std::vector<int>& indices{ pMesh->GetIndexes() };
		const Mesh::Textures& textures{ pMesh->GetTextures() };
		const Mesh::PrimitiveTopology pT{ pMesh->GetTopology() };
		const size_t size{ indices.size() };

		// Topology
		//"A switch statement does a hidden check to see if there are more than 10 things to check, not just if - else." -Terry A. Davis
		if (pT == Mesh::PrimitiveTopology::TriangleList)
		{
			for (size_t idx{ 2 }; idx < size; idx += 3)
			{
				const OVertex& v0{ NDCVertices[indices[idx - 2]] };
				const OVertex& v1{ NDCVertices[indices[idx - 1]] };
				const OVertex& v2{ NDCVertices[indices[idx]] };

				// separate triangle representation
				const OVertex triangle[3]{ v0, v1, v2 };

				if (!FrustumTest(triangle))
					continue;
				// FrustumTest complete

				SetVerticesToRasterScreenSpace(triangle); // transform vertices to raster/screen space
				uint32_t boundingValues[4]{};
				SetBoundingBox(boundingValues);
				RenderPixelsInTriangle(triangle, textures, boundingValues);
			}
		}
		else
		{
			bool isOdd{};//could replace with modulo operator, but this is way more performant, at the cost of using a tiny bit more memory
			//https://embeddedgurus.com/stack-overflow/2011/02/efficient-c-tip-13-use-the-modulus-operator-with-caution/
			for (size_t idx{}; idx < size - 2; ++idx)
			{
				const int idx0{ indices[idx] };
				const int idx1{ isOdd ? indices[idx + 2] : indices[idx + 1] };
				const int idx2{ isOdd ? indices[idx + 1] : indices[idx + 2] };

				const OVertex& v0{ NDCVertices[idx0] };
				const OVertex& v1{ NDCVertices[idx1] };
				const OVertex& v2{ NDCVertices[idx2] };

				// separate triangle representation
				const OVertex triangle[3]{ v0, v1, v2 };

				if (!FrustumTest(triangle))
					continue;

				SetVerticesToRasterScreenSpace(triangle);
				uint32_t boundingValues[4]{};
				SetBoundingBox(boundingValues);
				RenderPixelsInTriangle(triangle, textures, boundingValues);

				isOdd = !isOdd; // !flip last 2 vertices each odd triangle!
			}
		}
	}

	SDL_UnlockSurface(m_pBackBuffer);
	SDL_BlitSurface(m_pBackBuffer, 0, m_pFrontBuffer, 0);
	SDL_UpdateWindowSurface(m_pWindow);
}

bool Elite::Renderer::SaveBackbufferToImage() const
{
	return SDL_SaveBMP(m_pBackBuffer, "BackbufferRender.bmp");
}

OVertex Elite::Renderer::GetNDCVertex(const IVertex& vertex, const FMatrix4& worldMatrix)
{
	// viewmatrix
	//FMatrix4 viewMatrix{ std::move(Inverse(Camera::GetInstance()->GenerateLookAt())) };

	// projectionmatrix
	//FMatrix4 projectionMatrix{ std::move(Camera::GetInstance()->GetProjectionMatrix()) };

	// worldmatrix
	//FMatrix4 worldMatrix{ std::move(FMatrix4::Identity()) };

	// combined worldViewProjectionMatrix
	FMatrix4 worldViewProjectionMatrix = Camera::GetInstance()->GetProjectionMatrix() * Inverse(Camera::GetInstance()->GenerateLookAt()) * worldMatrix;

	FPoint4 NDCspace = worldViewProjectionMatrix * FPoint4{ vertex.v.x, vertex.v.y, vertex.v.z, vertex.v.z };

	// calculating vertex world position
	//FPoint3 worldLocation{ worldMatrix * FPoint4{ vertex.v } };

	FVector3 viewDirection{ GetNormalized(FPoint3{ worldMatrix * FPoint4{ vertex.v } } - Camera::GetInstance()->GetPos()) };

	// converting to NDCspace
	NDCspace.x /= NDCspace.w;
	NDCspace.y /= NDCspace.w;
	NDCspace.z /= NDCspace.w;
	//NDCspace.w = NDCspace.w;

	// converting to raster-/screenspace
	//NDCspace.x = ((NDCspace.x + 1) / 2) * m_Width;
	//NDCspace.y = ((1 - NDCspace.y) / 2) * m_Height;
	// !DONE AFTER FRUSTUMTEST!

	return { std::move(NDCspace), vertex.uv, std::move(FVector3{ worldMatrix * FVector4{vertex.n}}), std::move(FVector3{worldMatrix * FVector4{vertex.tan}}), vertex.c, std::move(viewDirection) };

	//----------------------------------------------------------------------------------------

	// aspectRatio and fov
	//float aspectRatio{ Camera::GetInstance()->GetAspectRatio() };
	//float fov{ Camera::GetInstance()->GetFov() };

	//{
	//	// view space
	//	FPoint3 viewSpaceVertex{ Inverse(Camera::GetInstance()->GenerateLookAt()) * FPoint4 { vertex.v } };

	//	// projection space (perspective divide)
	//	float projectedVertexX{ viewSpaceVertex.x / -viewSpaceVertex.z };
	//	projectedVertexX /= (aspectRatio * fov);

	//	float projectedVertexY{ viewSpaceVertex.y / -viewSpaceVertex.z };
	//	projectedVertexY /= fov;

	//	float projectedVertexZ{ -viewSpaceVertex.z };

	//	// screen space
	//	float screenspaceVertexX{ ((projectedVertexX + 1) / 2) * m_Width };
	//	float screenspaceVertexY{ ((1 - projectedVertexY) / 2) * m_Height };
	//	float screenspaceVertexZ{ projectedVertexZ };

	//	FPoint3 corrVertex{ screenspaceVertexX, screenspaceVertexY, screenspaceVertexZ };

	//	return OVertex{ corrVertex, vertex.uv, vertex.c };
	//}
}

std::vector<OVertex> Elite::Renderer::GetNDCMeshVertices(const std::vector<IVertex>& vertices, const FMatrix4& worldMatrix)
{
	std::vector<OVertex> corrVertices{};
	const size_t size{ vertices.size() };
	corrVertices.reserve(size);
	for (size_t i{}; i < size; ++i)
	{
		corrVertices.push_back(GetNDCVertex(vertices[i], worldMatrix));
	}
	return corrVertices;
	// DO NOT USE std::move() on return, since it will make corrVertices an rvalue, which cancels RVO (return value optimization)!
	// Basically the compilers optimizes the proces above so an automatic std::move() will be used instead of a copy construction (thank god)
}

void Elite::Renderer::RenderPixelsInTriangle(const OVertex triangle[3], const Mesh::Textures& textures)
{
	uint32_t boundingBox[4];
	SetBoundingBox(boundingBox);
	RenderPixelsInTriangle(triangle, textures, boundingBox);
}

void Elite::Renderer::RenderPixelsInTriangle(const OVertex triangle[3], const Mesh::Textures& textures, const uint32_t boundingValues[4])
{
	SceneManager& sm = *SceneManager::GetInstance();

	//Loop over all pixels in bounding box
	for (uint32_t r = boundingValues[1] - 1; r < boundingValues[3] + 1; ++r) // adding and subtracting 1 to get rid of seaming artifacts
	{
		for (uint32_t c = boundingValues[0] - 1; c < boundingValues[2] + 1; ++c)
		{
			const FPoint2 pixel{ float(c), float(r) };
			float weights[3]{}; // array of 3 addresses (pointers aa)

			if (IsPixelInTriangle(pixel, weights))
			{
				float zInterpolated{};
				if (DepthTest(triangle, m_pDepthBuffer[size_t(c) + (size_t(r) * m_Width)], weights, zInterpolated))
				{
					// apparently this method is faster
					FVector2 interpolatedUV{ weights[0] * (triangle[0].uv.x / triangle[0].v.w) + weights[1] * (triangle[1].uv.x / triangle[1].v.w) + weights[2] * (triangle[2].uv.x / triangle[2].v.w),
											 weights[0] * (triangle[0].uv.y / triangle[0].v.w) + weights[1] * (triangle[1].uv.y / triangle[1].v.w) + weights[2] * (triangle[2].uv.y / triangle[2].v.w) };
					//FVector2 interpolatedUV2{ (weights[0] * (triangle[0]->uv / triangle[0]->v.w)) + (weights[1] * (triangle[1]->uv / triangle[1]->v.w)) + (weights[2] * (triangle[2]->uv / triangle[2]->v.w)) };
					
					const float wInterpolated = (weights[0] * triangle[0].v.w) + (weights[1] * triangle[1].v.w) + (weights[2] * triangle[2].v.w);
					interpolatedUV *= wInterpolated;

					RGBColor finalColour{};

					if (!sm.IsDepthColour()) // show depth colour?
					{
						if (textures.pDiff) // diffuse map present?
						{
							const SampleState sampleState = sm.GetSampleState();
							const RGBColor diffuseColour = textures.pDiff->Sample(interpolatedUV, sampleState); // sample RGB colour

							if (textures.pNorm) // normal map present?
							{
								const RGBColor normalRGB = textures.pNorm->Sample(interpolatedUV, sampleState); // sample RGB Normal
								FVector3 normal{ normalRGB.r, normalRGB.g, normalRGB.b }; // sampled Normal form normalMap

								FVector3 interpolatedNormal{
									 weights[0] * (triangle[0].n.x / triangle[0].v.w) + weights[1] * (triangle[1].n.x / triangle[1].v.w) + weights[2] * (triangle[2].n.x / triangle[2].v.w),
									 weights[0] * (triangle[0].n.y / triangle[0].v.w) + weights[1] * (triangle[1].n.y / triangle[1].v.w) + weights[2] * (triangle[2].n.y / triangle[2].v.w),
									 weights[0] * (triangle[0].n.z / triangle[0].v.w) + weights[1] * (triangle[1].n.z / triangle[1].v.w) + weights[2] * (triangle[2].n.z / triangle[2].v.w) };
								// should be normalized anyway
								interpolatedNormal *= wInterpolated;

								const FVector3 interpolatedTangent{
									weights[0] * (triangle[0].tan.x / triangle[0].v.w) + weights[1] * (triangle[1].tan.x / triangle[1].v.w) + weights[2] * (triangle[2].tan.x / triangle[2].v.w),
									weights[0] * (triangle[0].tan.y / triangle[0].v.w) + weights[1] * (triangle[1].tan.y / triangle[1].v.w) + weights[2] * (triangle[2].tan.y / triangle[2].v.w),
									weights[0] * (triangle[0].tan.z / triangle[0].v.w) + weights[1] * (triangle[1].tan.z / triangle[1].v.w) + weights[2] * (triangle[2].tan.z / triangle[2].v.w) };
								// should be normalized from the parser

								FVector3 binormal{ Cross(interpolatedTangent, interpolatedNormal) };
								//Normalize(binormal);
								FMatrix3 tangentSpaceAxis{ interpolatedTangent, binormal, interpolatedNormal };

								//normal /= 255.f; // normal to [0, 1]
								normal.x = 2.f * normal.x - 1.f; // from [0, 1] to [-1, 1]
								normal.y = 2.f * normal.y - 1.f;
								normal.z = 2.f * normal.z - 1.f;
								// !Already defined in [0, 1]!

								normal = tangentSpaceAxis * normal; // normal defined in tangent space
								//Normalize(normal);

								// light calculations
								for (Light* pLight : sm.GetSceneGraph()->GetLights())
								{
									const FVector3& lightDir{ pLight->GetDirection(FPoint3{}) };
									const float observedArea{ Dot(-normal, lightDir) };

									if (observedArea < 0.f)
										continue;

									const RGBColor biradiance{ pLight->GetBiradiance(FPoint3{}) };
									// swapped direction of lights
									if (textures.pSpec && textures.pGloss) // specular and glossy map present?
									{
										FVector3 interpolatedViewDirection{
											weights[0] * (triangle[0].vd.y / triangle[0].v.w) + weights[1] * (triangle[1].vd.y / triangle[1].v.w) + weights[2] * (triangle[2].vd.y / triangle[2].v.w),
											weights[0] * (triangle[0].vd.x / triangle[0].v.w) + weights[1] * (triangle[1].vd.x / triangle[1].v.w) + weights[2] * (triangle[2].vd.x / triangle[2].v.w),
											weights[0] * (triangle[0].vd.z / triangle[0].v.w) + weights[1] * (triangle[1].vd.z / triangle[1].v.w) + weights[2] * (triangle[2].vd.z / triangle[2].v.w) };
										Normalize(interpolatedViewDirection);

										// phong
										const FVector3 reflectV{ Reflect(lightDir, normal) };
										//Normalize(reflectV);
										const float angle{ Dot(reflectV, interpolatedViewDirection) };
										const RGBColor specularSample{ textures.pSpec->Sample(interpolatedUV, sampleState) };
										const RGBColor phongSpecularReflection{ specularSample * powf(angle, textures.pGloss->Sample(interpolatedUV, sampleState).r * 25.f) };

										//const RGBColor lambertColour{ diffuseColour * (RGBColor{1.f,1.f,1.f} - specularSample) };
										//const RGBColor lambertColour{ (diffuseColour / float(E_PI)) * (RGBColor{1.f,1.f,1.f} - specularSample) };
										const RGBColor lambertColour{ (diffuseColour * specularSample) / float(E_PI) }; //severely incorrect result, using diffusecolour for now
										// Lambert diffuse == incoming colour multiplied by diffuse coefficient (1 in this case) divided by Pi
										finalColour += biradiance * (diffuseColour + phongSpecularReflection) * observedArea;
									}
									else
									{
										finalColour += biradiance * diffuseColour * observedArea;
										// without phong, with light(s)
									}
								}
								finalColour.Clamp();
							}
							else
							{
								//Without normals for the Tuk Tuk, this is impossible
								//for (Light* pLight : sm.GetSceneGraph()->GetLights())
								//{
								//	const float observedArea{ Dot(-n, pLight->GetDirection(FPoint3{})) };
								//	finalColour += pLight->GetBiradiance(FPoint3{}) * diffuseColour * observedArea;
								//}
								finalColour = diffuseColour;
							}
						}
						else
						{
							//without normals this is impossible, we'd need precalculated ones or calculate them on the go,
							//then proceed to transform them with their respective world matrix, which this loop doesn't support yet
							//for (Light* pLight : sm.GetSceneGraph()->GetLights())
							//{
							//	const float observedArea{ Dot(interpolatedNormal, pLight->GetDirection(FPoint3{})) };
							//	finalColour += pLight->GetBiradiance(FPoint3{}) * observedArea *
							//		(triangle[0].c * weights[0] + triangle[1].c * weights[1] + triangle[2].c * weights[2]);
							//}
							// else use plain colour
							finalColour = triangle[0].c * weights[0] + triangle[1].c * weights[1] + triangle[2].c * weights[2];
						}
						//finalColour.Clamp();
					}
					else
					{
						finalColour = RGBColor{ Remap(zInterpolated, 0.985f, 1.f), 0.f, 0.f }; // depth colour
						finalColour.Clamp();
					}

					// final draw
					m_pBackBufferPixels[c + (r * m_Width)] = SDL_MapRGB(m_pBackBuffer->format,
						static_cast<uint8_t>(finalColour.r * 255.f),
						static_cast<uint8_t>(finalColour.g * 255.f),
						static_cast<uint8_t>(finalColour.b * 255.f));
				}
			}
		}
	}
}

void Elite::Renderer::PixelShading(const FVector2& interpolatedUV)
{
}

bool Elite::Renderer::IsPixelInTriangle(const FPoint2& pixel, float weights[3])
{
	const FPoint2 v0{ m_RasterScreenSpaceX[0], m_RasterScreenSpaceY[0] };
	const FPoint2 v1{ m_RasterScreenSpaceX[1], m_RasterScreenSpaceY[1] };
	const FPoint2 v2{ m_RasterScreenSpaceX[2], m_RasterScreenSpaceY[2] };

	const FVector2 edgeA{ v0 - v1 };
	const FVector2 edgeB{ v1 - v2 };
	const FVector2 edgeC{ v2 - v0 };
	// counter-clockwise

	// edgeA
	FVector2 vertexToPixel{ pixel - v0 };
	const float a{ Cross(edgeA, vertexToPixel) };
	if (a < 0)
		return false;

	// edgeB
	vertexToPixel = { pixel - v1 };
	const float b{ Cross(edgeB, vertexToPixel) };
	if (b < 0)
		return false;

	// edgeC
	vertexToPixel = { pixel - v2 };
	const float c{ Cross(edgeC, vertexToPixel) };
	if (c < 0)
		return false;

	// weights
	const float totalArea{ Cross(edgeA, edgeC) };
	weights[0] = Cross(v1 - pixel, v1 - v2) / totalArea;
	weights[1] = Cross(v2 - pixel, v2 - v0) / totalArea;
	weights[2] = Cross(v0 - pixel, v0 - v1) / totalArea;
	// gives positive results because counter-clockwise

	//const float total = weights[0] + weights[1] + weights[2]; // total result equals 1

	return true;
}

bool Elite::Renderer::DepthTest(const OVertex triangle[3], float& depthBuffer, float weights[3], float& zInterpolated)
{
	zInterpolated = (weights[0] * triangle[0].v.z) + (weights[1] * triangle[1].v.z) + (weights[2] * triangle[2].v.z);
	//float wInterpolated = (weights[0] * triangle[0]->v.w) + (weights[1] * triangle[1]->v.w) + (weights[2] * triangle[2]->v.w);

	if (zInterpolated < 0 || zInterpolated > 1.f) return false;
	if (zInterpolated > depthBuffer) return false;

	depthBuffer = zInterpolated;

	return true;
}

bool Elite::Renderer::FrustumTestVertex(const OVertex& NDC)
{
	if (NDC.v.x < -1.f || NDC.v.x > 1.f) return false; // perspective divide X in NDC
	if (NDC.v.y < -1.f || NDC.v.y > 1.f) return false; // perspective divide Y in NDC
	if (NDC.v.z < 0.f || NDC.v.z > 1.f) return false; // perspective divide Z in NDC

	return true;
}

bool Elite::Renderer::FrustumTest(const OVertex NDC[3])
{
	if (!FrustumTestVertex(NDC[0])) return false;
	if (!FrustumTestVertex(NDC[1])) return false;
	return FrustumTestVertex(NDC[2]); //we can return the last check
}

void Elite::Renderer::SetVerticesToRasterScreenSpace(const OVertex triangle[3])
{
	for (int i{}; i < 3; ++i)
	{
		//triangle[i]->v.x = ((triangle[i]->v.x + 1) / 2) * m_Width;
		//triangle[i]->v.y = ((1 - triangle[i]->v.y) / 2) * m_Height;
		m_RasterScreenSpaceX[i] = ((triangle[i].v.x + 1) / 2) * m_Width;
		m_RasterScreenSpaceY[i] = ((1 - triangle[i].v.y) / 2) * m_Height;
	}
}

void Elite::Renderer::SetBoundingBox(uint32_t boundingValues[4])
{
	boundingValues[0] = uint32_t(std::min(std::initializer_list<float>{m_RasterScreenSpaceX[0], m_RasterScreenSpaceX[1], m_RasterScreenSpaceX[2]})); // xMin
	boundingValues[1] = uint32_t(std::min(std::initializer_list<float>{m_RasterScreenSpaceY[0], m_RasterScreenSpaceY[1], m_RasterScreenSpaceY[2]})); // yMin
	boundingValues[2] = uint32_t(std::max(std::initializer_list<float>{m_RasterScreenSpaceX[0], m_RasterScreenSpaceX[1], m_RasterScreenSpaceX[2]})); // xMax
	boundingValues[3] = uint32_t(std::max(std::initializer_list<float>{m_RasterScreenSpaceY[0], m_RasterScreenSpaceY[1], m_RasterScreenSpaceY[2]})); // yMax
	if (boundingValues[0] < 0) boundingValues[0] = 0;
	else if (boundingValues[0] > m_Width) boundingValues[0] = m_Width;
	if (boundingValues[1] < 0) boundingValues[1] = 0;
	else if (boundingValues[1] > m_Height) boundingValues[1] = m_Height;
	if (boundingValues[2] < 0) boundingValues[2] = 0;
	else if (boundingValues[2] > m_Width) boundingValues[2] = m_Width;
	if (boundingValues[3] < 0) boundingValues[3] = 0;
	else if (boundingValues[3] > m_Height) boundingValues[3] = m_Height;
}

void Elite::Renderer::CreateDepthBuffer()
{
	const size_t size{ m_Width * m_Height };
	if (!m_pDepthBuffer)
		m_pDepthBuffer = new float[size];
	std::memset(m_pDepthBuffer, INT_MAX, sizeof(float) * size);
}

void Elite::Renderer::ClearDepthBuffer()
{
	std::memset(m_pDepthBuffer, INT_MAX, sizeof(float) * m_Width * m_Height);//INT_MAX can be interpret as FLT_MAX bc of all bits being 1
}

void Elite::Renderer::ClearScreen()
{
	std::memset(m_pBackBufferPixels, 64, sizeof(uint8_t) * 4 * m_Width * m_Height);//times 4 bc RGBA, value of 64 for R,G,B for dark grey
}

void Elite::Renderer::BlackDraw(uint32_t c, uint32_t r)
{
	m_pBackBufferPixels[c + (r * m_Width)] = SDL_MapRGB(m_pBackBuffer->format,
		static_cast<uint8_t>(0.f),
		static_cast<uint8_t>(0.f),
		static_cast<uint8_t>(0.f));
}