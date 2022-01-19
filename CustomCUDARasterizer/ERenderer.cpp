#include "PCH.h"
//External includes
#include "SDL.h"
#include "SDL_surface.h"

//Project includes
#include "ERenderer.h"
#include <iostream>
#include "SceneManager.h"
#include "SceneGraph.h"
#include "Camera.h"
#include "Mesh.h"
#include "Texture.h"
#include "ObjParser.h"
#include "DirectionalLight.h"
#include "EventManager.h"
#include "BoundingBox.h"

Elite::Renderer::Renderer(SDL_Window* pWindow)
{
	m_pWindow = pWindow;
	m_pFrontBuffer = SDL_GetWindowSurface(pWindow);
	int width, height = 0;
	SDL_GetWindowSize(pWindow, &width, &height);
	m_Width = static_cast<uint32_t>(width);
	m_Height = static_cast<uint32_t>(height);
	m_pBackBuffer = SDL_CreateRGBSurface(0, m_Width, m_Height, 32, 0, 0, 0, 0);
	m_pBackBufferPixels = (uint32_t*)m_pBackBuffer->pixels;
	CreateDepthBuffer();
}

Elite::Renderer::~Renderer()
{
	if (m_pDepthBuffer)
		delete[] m_pDepthBuffer;
	m_pDepthBuffer = nullptr;
}

void Elite::Renderer::Render()
{
	//Create valid rendering state
	SDL_LockSurface(m_pBackBuffer);
	ClearDepthBuffer();
	ClearScreen();

	//SceneGraph
	SceneManager& sm = *SceneManager::GetInstance();
	SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pObjects = pSceneGraph->GetObjects();
	const FMatrix4 lookatMatrix = m_pCamera->GetLookAtMatrix();
	const FMatrix4 viewMatrix{ m_pCamera->GetViewMatrix(lookatMatrix) };
	const FMatrix4 projectionMatrix{ m_pCamera->GetProjectionMatrix() };
	const FMatrix4 viewProjectionMatrix = projectionMatrix * viewMatrix;

	//Render frame by looping over all objects
	for (Mesh* pMesh : pObjects)
	{
		std::vector<OVertex> NDCVertices{ GetNDCMeshVertices(pMesh->GetVertices(), viewProjectionMatrix, pMesh->GetWorldMatrix()) }; // calculate all IVertices to OVertices !in NDC!
		const std::vector<int>& indices{ pMesh->GetIndexes() };
		m_pTextures = &pMesh->GetTextures();
		const Mesh::PrimitiveTopology pT{ pMesh->GetTopology() };
		const size_t size{ indices.size() };

		// Topology
		//"A switch statement does a hidden check to see if there are more than 10 things to check, not just if - else." -Terry A. Davis
		if (pT == Mesh::PrimitiveTopology::TriangleList)
		{
			for (size_t idx{ 2 }; idx < size; idx += 3)
			{
				OVertex& v0 = NDCVertices[indices[idx - 2]];
				OVertex& v1 = NDCVertices[indices[idx - 1]];
				OVertex& v2 = NDCVertices[indices[idx]];

				// separate triangle representation (array of OVertex*)
				OVertex* NDCtriangle[3]{ &v0, &v1, &v2 };

				if (!FrustumTest(NDCtriangle))
					continue;

				RenderTriangle(NDCtriangle);
			}
		}
		else //if (pT == Mesh::PrimitiveTopology::TriangleStrip)
		{
			bool isOdd{};//could replace with modulo operator, but this is way more performant, at the cost of using a tiny bit more memory
			//https://embeddedgurus.com/stack-overflow/2011/02/efficient-c-tip-13-use-the-modulus-operator-with-caution/
			for (size_t idx{}; idx < size - 2; ++idx)
			{
				const int idx0 = indices[idx];
				const int idx1 = isOdd ? indices[idx + 2] : indices[idx + 1];
				const int idx2 = isOdd ? indices[idx + 1] : indices[idx + 2];

				OVertex& v0 = NDCVertices[idx0];
				OVertex& v1 = NDCVertices[idx1];
				OVertex& v2 = NDCVertices[idx2];

				// separate triangle representation (array of OVertex*)
				OVertex* triangle[3]{ &v0, &v1, &v2 };

				//if (!FrustumTest(triangle))
				//	continue;

				RenderTriangle(triangle);;

				isOdd = !isOdd; // !flip last 2 vertices each odd triangle!
			}
		}
	}

	//Swap out buffers to present new frame
	SDL_UnlockSurface(m_pBackBuffer);
	SDL_BlitSurface(m_pBackBuffer, 0, m_pFrontBuffer, 0);
	SDL_UpdateWindowSurface(m_pWindow);
}

void Elite::Renderer::RenderTriangle(OVertex* NDCTriangle[3])
{
	SetVerticesToRasterScreenSpace(NDCTriangle); //NDC to Screenspace
	RenderPixelsInTriangle(NDCTriangle); //Rasterize Screenspace triangle
}

bool Elite::Renderer::SaveBackbufferToImage() const
{
	return SDL_SaveBMP(m_pBackBuffer, "BackbufferRender.bmp");
}

OVertex Elite::Renderer::GetNDCVertexDeprecated(const IVertex& vertex, const FMatrix4& worldMatrix)
{
	//-----------------------------Vector Based-----------------------------

	// aspectRatio and fov
	const float aspectRatio{ m_pCamera->GetAspectRatio() };
	const float fov{ m_pCamera->GetFov() };

	// view space
	const FPoint3 viewSpaceVertex{ m_pCamera->GetViewMatrix() * FPoint4 { vertex.v } };

	// projection space (perspective divide)
	float projectedVertexX{ viewSpaceVertex.x / -viewSpaceVertex.z };
	projectedVertexX /= (aspectRatio * fov);

	float projectedVertexY{ viewSpaceVertex.y / -viewSpaceVertex.z };
	projectedVertexY /= fov;

	float projectedVertexZ{ -viewSpaceVertex.z };

	// screen space
	float screenspaceVertexX{ ((projectedVertexX + 1) / 2) * m_Width };
	float screenspaceVertexY{ ((1 - projectedVertexY) / 2) * m_Height };
	float screenspaceVertexZ{ projectedVertexZ };

	const FPoint3 corrVertex{ screenspaceVertexX, screenspaceVertexY, screenspaceVertexZ };

	const FPoint3 camPos = m_pCamera->GetPos();
	const FPoint3 worldPosition{ worldMatrix * FPoint4{ vertex.v } };
	const FVector3 viewDirection{ GetNormalized(worldPosition - camPos) };
	const FVector3 worldNormal{ worldMatrix * FVector4{ vertex.n } };
	const FVector3 worldTangent{ worldMatrix * FVector4{ vertex.tan } };

	return OVertex{ corrVertex,
		vertex.uv,
		worldNormal,
		worldTangent,
		vertex.c,
		viewDirection };
}

OVertex Elite::Renderer::GetNDCVertex(const IVertex& vertex, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix)
{
	//-----------------------------Matrix Based-----------------------------
	const FMatrix4 worldViewProjectionMatrix{ viewProjectionMatrix * worldMatrix };
	FPoint4 NDCspace = worldViewProjectionMatrix * FPoint4{ vertex.v.x, vertex.v.y, vertex.v.z, vertex.v.z };

	// converting to NDCspace
	NDCspace.x /= NDCspace.w;
	NDCspace.y /= NDCspace.w;
	NDCspace.z /= NDCspace.w;
	//NDCspace.w = NDCspace.w;

	// converting to raster-/screenspace
	//NDCspace.x = ((NDCspace.x + 1) / 2) * m_Width;
	//NDCspace.y = ((1 - NDCspace.y) / 2) * m_Height;
	// !DONE AFTER FRUSTUMTEST!

	const FPoint3 camPos = m_pCamera->GetPos();
	const FMatrix3 rotationMatrix = (FMatrix3)worldMatrix;
	const FPoint3 worldPosition{ worldMatrix * FPoint4{ vertex.v } };
	const FVector3 viewDirection{ GetNormalized(worldPosition - camPos) };
	const FVector3 worldNormal{ rotationMatrix * vertex.n };
	const FVector3 worldTangent{ rotationMatrix * vertex.tan };

	return OVertex{ NDCspace,
		vertex.uv,
		worldNormal,
		worldTangent,
		vertex.c,
		viewDirection };
}

std::vector<OVertex> Elite::Renderer::GetNDCMeshVertices(const std::vector<IVertex>& vertices, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix)
{
	std::vector<OVertex> corrVertices{};
	const size_t size{ vertices.size() };
	corrVertices.reserve(size);
	for (size_t i{}; i < size; ++i)
	{
		corrVertices.push_back(GetNDCVertex(vertices[i], viewProjectionMatrix, worldMatrix));
	}
	return corrVertices;
	// DO NOT USE std::move() on return, since it will make corrVertices an rvalue, which cancels RVO (return value optimization)!
	// Basically the compilers optimizes the proces above so an automatic std::move() will be used instead of a copy construction (thank god)
}

void Elite::Renderer::RenderPixelsInTriangle(OVertex* screenspaceTriangle[3])
{
	SceneManager& sm = *SceneManager::GetInstance();
	const BoundingBox bb = GetBoundingBox(screenspaceTriangle);

	const OVertex& v0 = *screenspaceTriangle[0];
	const OVertex& v1 = *screenspaceTriangle[1];
	const OVertex& v2 = *screenspaceTriangle[2];

	//Loop over all pixels in bounding box
	for (uint32_t r = bb.yMin; r < bb.yMax; ++r) // adding and subtracting 1 to get rid of seaming artifacts
	{
		for (uint32_t c = bb.xMin; c < bb.xMax; ++c)
		{
			const FPoint2 pixel{ float(c), float(r) };
			float weights[3]{}; // array of 3 addresses (pointers aa)

			if (IsPixelInTriangle(screenspaceTriangle, pixel, weights))
			{
				float zInterpolated{};
				if (DepthTest(screenspaceTriangle, m_pDepthBuffer[size_t(c) + (size_t(r) * m_Width)], weights, zInterpolated))
				{
					// apparently this method is faster
					FVector2 interpolatedUV{ weights[0] * (v0.uv.x / v0.v.w) + weights[1] * (v1.uv.x / v1.v.w) + weights[2] * (v2.uv.x / v2.v.w),
											 weights[0] * (v0.uv.y / v0.v.w) + weights[1] * (v1.uv.y / v1.v.w) + weights[2] * (v2.uv.y / v2.v.w) };
					//FVector2 interpolatedUV2{ (weights[0] * (triangle[0]->uv / triangle[0]->v.w)) + (weights[1] * (triangle[1]->uv / triangle[1]->v.w)) + (weights[2] * (triangle[2]->uv / triangle[2]->v.w)) };
					
					const float wInterpolated = (weights[0] * v0.v.w) + (weights[1] * v1.v.w) + (weights[2] * v2.v.w);
					interpolatedUV *= wInterpolated;

					RGBColor finalColour{};

					if (!sm.IsDepthColour()) // show depth colour?
					{
						if (m_pTextures->pDiff) // diffuse map present?
						{
							const SampleState sampleState = sm.GetSampleState();
							const RGBColor diffuseColour = m_pTextures->pDiff->Sample(interpolatedUV, sampleState); // sample RGB colour

							if (m_pTextures->pNorm) // normal map present?
							{
								const RGBColor normalRGB = m_pTextures->pNorm->Sample(interpolatedUV, sampleState); // sample RGB Normal
								FVector3 normal{ normalRGB.r, normalRGB.g, normalRGB.b }; // sampled Normal form normalMap

								FVector3 interpolatedNormal{
									 weights[0] * (v0.n.x / v0.v.w) + weights[1] * (v1.n.x / v1.v.w) + weights[2] * (v2.n.x / v2.v.w),
									 weights[0] * (v0.n.y / v0.v.w) + weights[1] * (v1.n.y / v1.v.w) + weights[2] * (v2.n.y / v2.v.w),
									 weights[0] * (v0.n.z / v0.v.w) + weights[1] * (v1.n.z / v1.v.w) + weights[2] * (v2.n.z / v2.v.w) };
								// should be normalized anyway
								interpolatedNormal *= wInterpolated;

								const FVector3 interpolatedTangent{
									weights[0] * (v0.tan.x / v0.v.w) + weights[1] * (v1.tan.x / v1.v.w) + weights[2] * (v2.tan.x / v2.v.w),
									weights[0] * (v0.tan.y / v0.v.w) + weights[1] * (v1.tan.y / v1.v.w) + weights[2] * (v2.tan.y / v2.v.w),
									weights[0] * (v0.tan.z / v0.v.w) + weights[1] * (v1.tan.z / v1.v.w) + weights[2] * (v2.tan.z / v2.v.w) };
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

								FVector3 interpolatedViewDirection{
								weights[0] * (v0.vd.y / v1.v.w) + weights[1] * (v1.vd.y / v1.v.w) + weights[2] * (v2.vd.y / v2.v.w),
								weights[0] * (v0.vd.x / v1.v.w) + weights[1] * (v1.vd.x / v1.v.w) + weights[2] * (v2.vd.x / v2.v.w),
								weights[0] * (v0.vd.z / v1.v.w) + weights[1] * (v1.vd.z / v1.v.w) + weights[2] * (v2.vd.z / v2.v.w) };
								Normalize(interpolatedViewDirection);

								//OVertex oVertex{};
								//oVertex.v = FPoint4{ pixel, zInterpolated, wInterpolated };
								//oVertex.uv = interpolatedUV;
								//oVertex.n = interpolatedNormal;
								//oVertex.tan = interpolatedTangent;
								//oVertex.vd = interpolatedViewDirection;
								//ShadePixel(oVertex, textures);

								// light calculations
								for (Light* pLight : sm.GetSceneGraph()->GetLights())
								{
									const FVector3& lightDir{ pLight->GetDirection(FPoint3{}) };
									const float observedArea{ Dot(-normal, lightDir) };

									if (observedArea < 0.f)
										continue;

									const RGBColor biradiance{ pLight->GetBiradiance(FPoint3{}) };
									// swapped direction of lights
									if (m_pTextures->pSpec && m_pTextures->pGloss) // specular and glossy map present?
									{
										// phong
										const FVector3 reflectV{ Reflect(lightDir, normal) };
										//Normalize(reflectV);
										const float angle{ Dot(reflectV, interpolatedViewDirection) };
										const RGBColor specularSample{ m_pTextures->pSpec->Sample(interpolatedUV, sampleState) };
										const RGBColor phongSpecularReflection{ specularSample * powf(angle, m_pTextures->pGloss->Sample(interpolatedUV, sampleState).r * 25.f) };

										//const RGBColor lambertColour{ diffuseColour * (RGBColor{1.f,1.f,1.f} - specularSample) };
										//const RGBColor lambertColour{ (diffuseColour / float(E_PI)) * (RGBColor{1.f,1.f,1.f} - specularSample) };
										const RGBColor lambertColour{ (diffuseColour * specularSample) / float(PI) }; //severely incorrect result, using diffusecolour for now
										// Lambert diffuse == incoming colour multiplied by diffuse coefficient (1 in this case) divided by Pi
										finalColour += biradiance * (diffuseColour + phongSpecularReflection) * observedArea;
									}
									else
									{
										finalColour += biradiance * diffuseColour * observedArea;
										// without phong, with light(s)
									}
								}
								finalColour.ClampColor();
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
							finalColour = v0.c * weights[0] + v1.c * weights[1] + v2.c * weights[2];
						}
						//finalColour.Clamp();
					}
					else
					{
						finalColour = RGBColor{ Remap(zInterpolated, 0.985f, 1.f), 0.f, 0.f }; // depth colour
						finalColour.ClampColor();
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

void Elite::Renderer::ShadePixel(const OVertex& oVertex, const Mesh::Textures& textures)
{
	SceneManager& sm = *SceneManager::GetInstance();
	
	RGBColor finalColour{};
	
	if (!sm.IsDepthColour()) // show depth colour?
	{
		const SampleState sampleState = sm.GetSampleState();
		const RGBColor diffuseColour = textures.pDiff->Sample(oVertex.uv, sampleState); // sample RGB colour
	
		const RGBColor normalRGB = textures.pNorm->Sample(oVertex.uv, sampleState); // sample RGB Normal
		FVector3 normal{ normalRGB.r, normalRGB.g, normalRGB.b }; // sampled Normal form normalMap
	
		FVector3 binormal{ Cross(oVertex.tan, oVertex.n) };
		//Normalize(binormal);
		FMatrix3 tangentSpaceAxis{ oVertex.tan, binormal, oVertex.n };
	
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
	
			// phong
			const FVector3 reflectV{ Reflect(lightDir, normal) };
			//Normalize(reflectV);
			const float angle{ Dot(reflectV, oVertex.vd) };
			const RGBColor specularSample{ textures.pSpec->Sample(oVertex.uv, sampleState) };
			const RGBColor phongSpecularReflection{ specularSample * powf(angle, textures.pGloss->Sample(oVertex.uv, sampleState).r * 25.f) };
	
			//const RGBColor lambertColour{ diffuseColour * (RGBColor{1.f,1.f,1.f} - specularSample) };
			//const RGBColor lambertColour{ (diffuseColour / float(E_PI)) * (RGBColor{1.f,1.f,1.f} - specularSample) };
			const RGBColor lambertColour{ (diffuseColour * specularSample) / float(PI) }; //severely incorrect result, using diffusecolour for now
			// Lambert diffuse == incoming colour multiplied by diffuse coefficient (1 in this case) divided by Pi
			finalColour += biradiance * (diffuseColour + phongSpecularReflection) * observedArea;
		}
		finalColour.ClampColor();
	}
	else
	{
		finalColour = RGBColor{ Remap(oVertex.v.z, 0.985f, 1.f), 0.f, 0.f }; // depth colour
		finalColour.ClampColor();
	}
	
	// final draw
	m_pBackBufferPixels[(int)oVertex.v.x + (int)(oVertex.v.y * m_Width)] = SDL_MapRGB(m_pBackBuffer->format,
		static_cast<uint8_t>(finalColour.r * 255.f),
		static_cast<uint8_t>(finalColour.g * 255.f),
		static_cast<uint8_t>(finalColour.b * 255.f));
}

bool Elite::Renderer::IsPixelInTriangle(OVertex* screenspaceTriangle[3], const FPoint2& pixel, float weights[3])
{
	const FPoint2& v0 = screenspaceTriangle[0]->v.xy;
	const FPoint2& v1 = screenspaceTriangle[1]->v.xy;
	const FPoint2& v2 = screenspaceTriangle[2]->v.xy;

	const FVector2 edgeA{ v0 - v1 };
	const FVector2 edgeB{ v1 - v2 };
	const FVector2 edgeC{ v2 - v0 };
	// counter-clockwise

	const float totalArea = Cross(edgeA, edgeC);

	// edgeA
	FVector2 vertexToPixel{ pixel - v0 };
	float cross = Cross(edgeA, vertexToPixel);
	if (cross < 0.f)
		return false;
	// weight2 == positive cross of 'previous' edge (for v2 this is edgeA)
	weights[2] = cross / totalArea;

	// edgeB
	vertexToPixel = { pixel - v1 };
	cross = Cross(edgeB, vertexToPixel);
	if (cross < 0.f)
		return false;
	// weight1
	weights[1] = cross / totalArea;

	// edgeC
	vertexToPixel = { pixel - v2 };
	cross = Cross(edgeC, vertexToPixel);
	if (cross < 0.f)
		return false;
	// weight0
	weights[0] = cross / totalArea;

	//weights == inverted negative cross of 'previous' edge
	//weights[0] = Cross(-vertexToPixel, edgeC) / totalArea;
	//weights[1] = Cross(-vertexToPixel, edgeB) / totalArea;
	//weights[2] = Cross(-vertexToPixel, edgeA) / totalArea;
	// gives positive results because counter-clockwise
	//const float total = weights[0] + weights[1] + weights[2]; // total result equals 1

	return true;
}

bool Elite::Renderer::DepthTest(OVertex* triangle[3], float& depthBuffer, float weights[3], float& zInterpolated)
{
	zInterpolated = (weights[0] * triangle[0]->v.z) + (weights[1] * triangle[1]->v.z) + (weights[2] * triangle[2]->v.z);
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

bool Elite::Renderer::FrustumTest(OVertex* NDC[3])
{
	if (!FrustumTestVertex(*NDC[0])) return false;
	if (!FrustumTestVertex(*NDC[1])) return false;
	return FrustumTestVertex(*NDC[2]); //we can return the last check
}

void Elite::Renderer::SetVerticesToRasterScreenSpace(OVertex* triangle[3])
{
	for (int i{}; i < 3; ++i)
	{
		triangle[i]->v.x = ((triangle[i]->v.x + 1) / 2) * m_Width;
		triangle[i]->v.y = ((triangle[i]->v.y + 1) / 2) * m_Height;
	}
}

BoundingBox Elite::Renderer::GetBoundingBox(OVertex* screenspaceTriangle[3])
{
	//TODO: make member variable and update
	BoundingBox bb;
	bb.xMin = unsigned short(std::min(std::initializer_list<float>{screenspaceTriangle[0]->v.x, screenspaceTriangle[1]->v.x, screenspaceTriangle[2]->v.x})) - 1; // xMin
	bb.yMin = unsigned short(std::min(std::initializer_list<float>{screenspaceTriangle[0]->v.y, screenspaceTriangle[1]->v.y, screenspaceTriangle[2]->v.y})) - 1; // yMin
	bb.xMax = unsigned short(std::max(std::initializer_list<float>{screenspaceTriangle[0]->v.x, screenspaceTriangle[1]->v.x, screenspaceTriangle[2]->v.x})) + 1; // xMax
	bb.yMax = unsigned short(std::max(std::initializer_list<float>{screenspaceTriangle[0]->v.y, screenspaceTriangle[1]->v.y, screenspaceTriangle[2]->v.y})) + 1; // yMax

	if (bb.xMin < 0) bb.xMin = 0; //clamp minX to Left of screen
	if (bb.yMin < 0) bb.yMin = 0; //clamp minY to Bottom of screen
	if (bb.xMax > m_Width) bb.xMax = m_Width; //clamp maxX to Right of screen
	if (bb.yMax > m_Height) bb.yMax = m_Height; //clamp maxY to Top of screen

	return bb;
}

void Elite::Renderer::CreateDepthBuffer()
{
	if (!m_pDepthBuffer)
		m_pDepthBuffer = new float[m_Width * m_Height];
	ClearDepthBuffer();
}

void Elite::Renderer::ClearDepthBuffer()
{
	std::memset(m_pDepthBuffer, INT_MAX, sizeof(float) * m_Width * m_Height);
}

void Elite::Renderer::ClearScreen()
{
	const int colourValue = 64; //value of 64 for R,G,B colour of dark grey
	std::memset(m_pBackBufferPixels, colourValue, sizeof(SDL_Color) * m_Width * m_Height);
	//SDL_Color => RGBA (4 * Uint8)
}

void Elite::Renderer::BlackDraw(unsigned short c, unsigned short r)
{
	m_pBackBufferPixels[c + (r * m_Width)] = SDL_MapRGB(m_pBackBuffer->format, 0, 0, 0);
}