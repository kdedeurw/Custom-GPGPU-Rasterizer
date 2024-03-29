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
#include "Textures.h"
#include "ObjParser.h"
#include "DirectionalLight.h"
#include "EventManager.h"
#include "BoundingBox.h"
#include "PrimitiveTopology.h"
#include "CullingMode.h"

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

void Elite::Renderer::Render(const SceneManager& sm)
{
	//Create valid rendering state
	SDL_LockSurface(m_pBackBuffer);
	ClearDepthBuffer();
	ClearScreen();

	//SceneGraph
	SceneGraph* pSceneGraph = sm.GetSceneGraph();
	const std::vector<Mesh*>& pObjects = pSceneGraph->GetMeshes();
	const FVector3& camFwd = m_pCamera->GetForward();
	//const FMatrix4& lookatMatrix = m_pCamera->GetLookAtMatrix();
	const FMatrix4 viewMatrix = m_pCamera->GetViewMatrix();
	const FMatrix4& projectionMatrix = m_pCamera->GetProjectionMatrix();
	const FMatrix4 viewProjectionMatrix = projectionMatrix * viewMatrix;

	//Render frame by looping over all objects
	for (Mesh* pMesh : pObjects)
	{
		const std::vector<IVertex>& vertexBuffer = pMesh->GetVertexBuffer();
		const unsigned int numVertices = pMesh->GetVertexAmount();
		const std::vector<unsigned int>& indexBuffer = pMesh->GetIndexBuffer();
		const unsigned int numIndices = pMesh->GetIndexAmount();
		m_pTextures = &pMesh->GetTextures();
		const PrimitiveTopology pT{ pMesh->GetTopology() };
		const CullingMode cm{ sm.GetCullingMode() };

		std::vector<OVertex> NDCVertices{ GetNDCMeshVertices(vertexBuffer, viewProjectionMatrix, pMesh->GetWorldMatrix()) }; // calculate all IVertices to OVertices !in NDC!

		// Topology
		//"A switch statement does a hidden check to see if there are more than 10 things to check, not just if - else." -Terry A. Davis
		if (pT == PrimitiveTopology::TriangleList)
		{
			for (size_t idx{ 2 }; idx < numIndices; idx += 3)
			{
				const OVertex& v0 = NDCVertices[indexBuffer[idx - 2]];
				const OVertex& v1 = NDCVertices[indexBuffer[idx - 1]];
				const OVertex& v2 = NDCVertices[indexBuffer[idx]];

				//is triangle visible according to cullingmode?
				if (cm == CullingMode::BackFace)
				{
					const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
					const float cullingValue = Dot(camFwd, faceNormal);
					//is back facing triangle?
					if (cullingValue <= 0.f)
						continue;
				}
				else if (cm == CullingMode::FrontFace)
				{
					//TODO: render entire object but invert all triangles
					//const OVertex& v0Inv = NDCVertices[indices[idx]];
					//const OVertex& v1Inv = NDCVertices[indices[idx - 1]];
					//const OVertex& v2Inv = NDCVertices[indices[idx - 2]];

					const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
					const float cullingValue = Dot(camFwd, faceNormal);
					////is front facing triangle?
					if (cullingValue >= 0.f)
						continue;
				}

				// separate triangle representation (array of OVertex*)
				const OVertex* triangle[3]{ &v0, &v1, &v2 };
				FPoint4 rasterCoords[3]{ v0.p, v1.p, v2.p }; //painful, but unavoidable copy
				//Otherwise any mesh that uses a vertex twice will literally get shredded due to same values being used for frustum tests etc.

				if (!IsTriangleVisible(rasterCoords))
					continue;

				RenderTriangle(sm, triangle, rasterCoords);
			}
		}
		else //if (pT == PrimitiveTopology::TriangleStrip)
		{
			bool isOdd{};//could replace with modulo operator, but this is way more performant, at the cost of using a tiny bit more memory
			//https://embeddedgurus.com/stack-overflow/2011/02/efficient-c-tip-13-use-the-modulus-operator-with-caution/
			for (size_t idx{}; idx < numIndices - 2; ++idx)
			{
				const unsigned int idx0 = indexBuffer[idx];
				const unsigned int idx1 = isOdd ? indexBuffer[idx + 1] : indexBuffer[idx + 2];
				const unsigned int idx2 = isOdd ? indexBuffer[idx + 2] : indexBuffer[idx + 1];

				const OVertex& v0 = NDCVertices[idx0];
				const OVertex& v1 = NDCVertices[idx1];
				const OVertex& v2 = NDCVertices[idx2];

				//is triangle visible according to cullingmode?
				if (cm == CullingMode::BackFace)
				{
					const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
					//is back facing triangle?
					if (Dot(camFwd, faceNormal) <= 0.f)
						continue;
				}
				else if (cm == CullingMode::FrontFace)
				{
					const FVector3 faceNormal = GetNormalized(Cross(FVector3{ v1.p - v0.p }, FVector3{ v2.p - v0.p }));
					//is front facing triangle?
					if (Dot(camFwd, faceNormal) >= 0.f)
						continue;
				}

				// separate triangle representation (array of OVertex*)
				const OVertex* triangle[3]{ &v0, &v1, &v2 };
				FPoint4 rasterCoords[3]{ v0.p, v1.p, v2.p }; //painful, but unavoidable copy
				//Otherwise any mesh that uses a vertex twice will literally get shredded due to same values being used for frustum tests etc.

				if (!IsTriangleVisible(rasterCoords))
					continue;

				RenderTriangle(sm, triangle, rasterCoords);

				isOdd = !isOdd; // !flip last 2 vertices each odd triangle!
			}
		}
	}

	//Swap out buffers to present new frame
	SDL_UnlockSurface(m_pBackBuffer);
	SDL_BlitSurface(m_pBackBuffer, 0, m_pFrontBuffer, 0);
	SDL_UpdateWindowSurface(m_pWindow);
}

void Elite::Renderer::RenderTriangle(const SceneManager& sm, const OVertex* triangle[3], FPoint4 rasterCoords[3])
{
	NDCToScreenSpace(rasterCoords[0]); //NDC to Screenspace
	NDCToScreenSpace(rasterCoords[1]); //NDC to Screenspace
	NDCToScreenSpace(rasterCoords[2]); //NDC to Screenspace
	RenderPixelsInTriangle(sm, triangle, rasterCoords); //Rasterize Screenspace triangle
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
	const FPoint3 viewSpaceVertex{ m_pCamera->GetViewMatrix() * FPoint4 { vertex.p } };

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

	const FPoint3& camPos = m_pCamera->GetPosition();
	const FPoint3 worldPosition{ worldMatrix * FPoint4{ vertex.p } };
	const FVector3 viewDirection{ GetNormalized(worldPosition - camPos) };
	const FVector3 worldNormal{ (FMatrix3)worldMatrix * vertex.n };
	const FVector3 worldTangent{ (FMatrix3)worldMatrix * vertex.tan };

	return OVertex{ corrVertex, vertex.uv, worldNormal, worldTangent, vertex.c, viewDirection };
}

OVertex Elite::Renderer::GetNDCVertex(const IVertex& vertex, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix)
{
	//-----------------------------Matrix Based-----------------------------
	const FMatrix4 worldViewProjectionMatrix{ viewProjectionMatrix * worldMatrix };
	const FMatrix3 rotationMatrix = (FMatrix3)worldMatrix;
	//FPoint4 NDCspace = worldViewProjectionMatrix * FPoint4{ vertex.p.x, vertex.p.y, vertex.p.z, vertex.p.z };
	FPoint4 NDCspace = worldViewProjectionMatrix * FPoint4{ vertex.p };

	// converting to NDCspace
	NDCspace.x /= NDCspace.w;
	NDCspace.y /= NDCspace.w;
	NDCspace.z /= NDCspace.w;
	//NDCspace.w = NDCspace.w;

	// converting to raster-/screenspace
	//NDCspace.x = ((NDCspace.x + 1) / 2) * m_Width;
	//NDCspace.y = ((1 - NDCspace.y) / 2) * m_Height;
	// !DONE AFTER FRUSTUMTEST!

	const FPoint3 camPos = m_pCamera->GetPosition();
	//const FMatrix3 rotationMatrix = (FMatrix3)worldMatrix;
	const FPoint3 worldPosition{ worldMatrix * FPoint4{ vertex.p } };
	const FVector3 viewDirection{ GetNormalized(worldPosition - camPos) };
	const FVector3 worldNormal{ rotationMatrix * vertex.n };
	const FVector3 worldTangent{ rotationMatrix * vertex.tan };

	return OVertex{ NDCspace, vertex.uv, worldNormal, worldTangent, vertex.c, viewDirection };
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

std::vector<OVertex> Elite::Renderer::GetNDCMeshVertices(const float* pVertices, unsigned int numVertices, const FMatrix4& viewProjectionMatrix, const FMatrix4& worldMatrix)
{
	std::vector<OVertex> corrVertices{};
	corrVertices.reserve(numVertices);
	for (size_t i{}; i < numVertices; ++i)
	{
		corrVertices.push_back(GetNDCVertex(reinterpret_cast<const IVertex*>(pVertices)[i], viewProjectionMatrix, worldMatrix));
	}
	return corrVertices;
}

void Elite::Renderer::RenderPixelsInTriangle(const SceneManager& sm, const OVertex* triangle[3], FPoint4 rasterCoords[3])
{
	const BoundingBox bb = GetBoundingBox(rasterCoords);

	const OVertex& v0 = *triangle[0];
	const OVertex& v1 = *triangle[1];
	const OVertex& v2 = *triangle[2];

	const float v0InvDepth = 1.f / rasterCoords[0].w;
	const float v1InvDepth = 1.f / rasterCoords[1].w;
	const float v2InvDepth = 1.f / rasterCoords[2].w;

	//Loop over all pixels in bounding box
	for (short r = bb.yMin; r < bb.yMax; ++r)
	{
		for (short c = bb.xMin; c < bb.xMax; ++c)
		{
			const FPoint2 pixel{ float(c), float(r) };
			float weights[3]{}; // array of 3 addresses

			if (IsPixelInTriangle(rasterCoords, pixel, weights))
			{
				float zInterpolated{};
				const size_t pixelId = c + r * m_Width;
				if (DepthTest(rasterCoords, m_pDepthBuffer[pixelId], weights, zInterpolated))
				{
					const float wInterpolated = 1.f / (v0InvDepth * weights[0] + v1InvDepth * weights[1] + v2InvDepth * weights[2]);

					FVector2 interpolatedUV{
						weights[0] * (v0.uv.x * v0InvDepth) + weights[1] * (v1.uv.x * v1InvDepth) + weights[2] * (v2.uv.x * v2InvDepth),
						weights[0] * (v0.uv.y * v0InvDepth) + weights[1] * (v1.uv.y * v1InvDepth) + weights[2] * (v2.uv.y * v2InvDepth) };
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
									 weights[0] * (v0.n.x * v0InvDepth) + weights[1] * (v1.n.x * v1InvDepth) + weights[2] * (v2.n.x * v2InvDepth),
									 weights[0] * (v0.n.y * v0InvDepth) + weights[1] * (v1.n.y * v1InvDepth) + weights[2] * (v2.n.y * v2InvDepth),
									 weights[0] * (v0.n.z * v0InvDepth) + weights[1] * (v1.n.z * v1InvDepth) + weights[2] * (v2.n.z * v2InvDepth) };
								// should be normalized anyway
								interpolatedNormal *= wInterpolated;

								const FVector3 interpolatedTangent{
									weights[0] * (v0.tan.x * v0InvDepth) + weights[1] * (v1.tan.x * v1InvDepth) + weights[2] * (v2.tan.x * v2InvDepth),
									weights[0] * (v0.tan.y * v0InvDepth) + weights[1] * (v1.tan.y * v1InvDepth) + weights[2] * (v2.tan.y * v2InvDepth),
									weights[0] * (v0.tan.z * v0InvDepth) + weights[1] * (v1.tan.z * v1InvDepth) + weights[2] * (v2.tan.z * v2InvDepth) };
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
									weights[0] * (v0.vd.x * v0InvDepth) + weights[1] * (v1.vd.x * v1InvDepth) + weights[2] * (v2.vd.x * v2InvDepth),
									weights[0] * (v0.vd.y * v0InvDepth) + weights[1] * (v1.vd.y * v1InvDepth) + weights[2] * (v2.vd.y * v2InvDepth),
									weights[0] * (v0.vd.z * v0InvDepth) + weights[1] * (v1.vd.z * v1InvDepth) + weights[2] * (v2.vd.z * v2InvDepth) };
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
					m_pBackBufferPixels[pixelId] = SDL_MapRGB(m_pBackBuffer->format,
						static_cast<uint8_t>(finalColour.r * 255.f),
						static_cast<uint8_t>(finalColour.g * 255.f),
						static_cast<uint8_t>(finalColour.b * 255.f));
				}
			}
		}
	}
}

void Elite::Renderer::ShadePixel(const OVertex& oVertex, const Textures& textures, const SceneManager& sm)
{
	RGBColor finalColour{};
	if (!sm.IsDepthColour()) // show depth colour?
	{
		const SampleState sampleState = sm.GetSampleState();

		//global settings
		bool isFlipGreenChannel = false;
		const float shininess = 25.f;
		const RGBColor ambientColour{ 0.05f, 0.05f, 0.05f };

		// texture sampling
		const RGBColor diffuseSample = textures.pDiff->Sample(oVertex.uv, sampleState);
		const RGBColor normalSample = textures.pNorm->Sample(oVertex.uv, sampleState);
		const RGBColor specularSample = textures.pSpec->Sample(oVertex.uv, sampleState);
		const RGBColor glossSample = textures.pGloss->Sample(oVertex.uv, sampleState);

		// normal mapping
		FVector3 binormal = Cross(oVertex.tan, oVertex.n);
		if (isFlipGreenChannel)
			binormal = -binormal;
		const FMatrix3 tangentSpaceAxis{ oVertex.tan, binormal, oVertex.n };

		FVector3 finalNormal{ 2.f * normalSample.r - 1.f, 2.f * normalSample.g - 1.f, 2.f * normalSample.b - 1.f };
		finalNormal = tangentSpaceAxis * finalNormal;

		// light calculations
		float totalObservedArea{};
		float totalAngle{};
		for (Light* pLight : sm.GetSceneGraph()->GetLights())
		{
			const FVector3& lightDirection = pLight->GetDirection({});
			float observedArea{ Dot(-finalNormal, lightDirection) };
			Clamp(observedArea, 0.f, observedArea);
			observedArea /= (float)PI;
			observedArea *= pLight->GetIntensity();
			totalObservedArea += observedArea;

			// phong specular
			const FVector3 reflectV{ Reflect(lightDirection, finalNormal) };
			float angle{ Dot(oVertex.vd, reflectV) }; //MIGHT SWAP THESE
			Clamp(angle, 0.f, 1.f);
			angle = powf(angle, glossSample.r * shininess);
			totalAngle += angle; //might clamp this
		}

		// final
		const RGBColor diffuseColour = diffuseSample * totalObservedArea;
		const RGBColor specularColour = specularSample * totalAngle;
		finalColour = ambientColour + diffuseColour + specularColour;
	}
	else
	{
		finalColour = RGBColor{ Remap(oVertex.p.z, 0.985f, 1.f), 0.f, 0.f }; // depth colour
	}
	finalColour.ClampColor();
	
	// final draw
	m_pBackBufferPixels[(int)oVertex.p.x + (int)(oVertex.p.y * m_Width)] = SDL_MapRGB(m_pBackBuffer->format,
		static_cast<uint8_t>(finalColour.r * 255.f),
		static_cast<uint8_t>(finalColour.g * 255.f),
		static_cast<uint8_t>(finalColour.b * 255.f));
}

bool Elite::Renderer::IsPixelInTriangle(FPoint4 rasterCoords[3], const FPoint2& pixel, float weights[3])
{
	const FPoint2& v0 = rasterCoords[0].xy;
	const FPoint2& v1 = rasterCoords[1].xy;
	const FPoint2& v2 = rasterCoords[2].xy;

	const FVector2 edgeA{ v0 - v1 };
	const FVector2 edgeB{ v1 - v2 };
	const FVector2 edgeC{ v2 - v0 };
	// counter-clockwise

	const float totalArea = abs(Cross(edgeA, edgeC));

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
	weights[0] = cross / totalArea;

	// edgeC
	vertexToPixel = { pixel - v2 };
	cross = Cross(edgeC, vertexToPixel);
	if (cross < 0.f)
		return false;
	// weight0
	weights[1] = cross / totalArea;

	//weights == inverted negative cross of 'previous' edge
	//weights[1] = Cross(-vertexToPixel, edgeC) / totalArea;
	//weights[0] = Cross(-vertexToPixel, edgeB) / totalArea;
	//weights[2] = Cross(-vertexToPixel, edgeA) / totalArea;
	// gives positive results because counter-clockwise
	//const float total = weights[0] + weights[1] + weights[2]; // total result equals 1

	return true;
}

bool Elite::Renderer::DepthTest(FPoint4 rasterCoords[3], float& depthBuffer, float weights[3], float& zInterpolated)
{
	zInterpolated = (weights[0] * rasterCoords[0].z) + (weights[1] * rasterCoords[1].z) + (weights[2] * rasterCoords[2].z);

	if (zInterpolated < 0 || zInterpolated > 1.f) return false;
	if (zInterpolated > depthBuffer) return false;

	depthBuffer = zInterpolated;

	return true;
}

bool Elite::Renderer::IsAllXOutsideFrustum(FPoint4 NDC[3]) const
{
	return	(NDC[0].x < -1.f && NDC[1].x < -1.f && NDC[2].x < -1.f) ||
		(NDC[0].x > 1.f && NDC[1].x > 1.f && NDC[2].x > 1.f);
}

bool Elite::Renderer::IsAllYOutsideFrustum(FPoint4 NDC[3]) const
{
	return	(NDC[0].y < -1.f && NDC[1].y < -1.f && NDC[2].y < -1.f) ||
		(NDC[0].y > 1.f && NDC[1].y > 1.f && NDC[2].y > 1.f);
}

bool Elite::Renderer::IsAllZOutsideFrustum(FPoint4 NDC[3]) const
{
	return	(NDC[0].z < 0.f && NDC[1].z < 0.f && NDC[2].z < 0.f) ||
		(NDC[0].z > 1.f && NDC[1].z > 1.f && NDC[2].z > 1.f);
}

bool Elite::Renderer::IsTriangleVisible(FPoint4 NDC[3]) const
{
	// Solution to FrustumCulling bug
	//	   if (all x values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all y values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all z values are < 0.f or > 1.f) AT ONCE, cull
	if (IsAllXOutsideFrustum(NDC)) return false;
	if (IsAllYOutsideFrustum(NDC)) return false;
	if (IsAllZOutsideFrustum(NDC)) return false;
	return true;
}

bool Elite::Renderer::IsVertexInFrustum(const FPoint4& NDC) const
{
	//If the vertex is outside of the frustum
	if (NDC.x < -1.f || NDC.x > 1.f) return false; // perspective divide X in NDC
	if (NDC.y < -1.f || NDC.y > 1.f) return false; // perspective divide Y in NDC
	if (NDC.z < 0.f || NDC.z > 1.f) return false; // perspective divide Z in NDC
	return true;
}

bool Elite::Renderer::IsTriangleInFrustum(FPoint4 rasterCoords[3]) const
{
	//If any of the vertices are inside of the frustum
	return(IsVertexInFrustum(rasterCoords[0]) 
		|| IsVertexInFrustum(rasterCoords[1]) 
		|| IsVertexInFrustum(rasterCoords[2]));
}

void Elite::Renderer::NDCToScreenSpace(FPoint4& rasterCoords)
{
	rasterCoords.x = ((rasterCoords.x + 1) / 2) * m_Width;
	rasterCoords.y = ((1 - rasterCoords.y) / 2) * m_Height;
}

BoundingBox Elite::Renderer::GetBoundingBox(FPoint4 rasterCoords[3])
{
	//TODO: make member variable and update
	BoundingBox bb;
	bb.xMin = short(std::min(std::initializer_list<float>{rasterCoords[0].x, rasterCoords[1].x, rasterCoords[2].x})) - 1; // xMin
	bb.yMin = short(std::min(std::initializer_list<float>{rasterCoords[0].y, rasterCoords[1].y, rasterCoords[2].y})) - 1; // yMin
	bb.xMax = short(std::max(std::initializer_list<float>{rasterCoords[0].x, rasterCoords[1].x, rasterCoords[2].x})) + 1; // xMax
	bb.yMax = short(std::max(std::initializer_list<float>{rasterCoords[0].y, rasterCoords[1].y, rasterCoords[2].y})) + 1; // yMax

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