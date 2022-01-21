#include "PCH.h"
#include "CUDAROPs.cuh"

#include "Vertex.h"
#include "Camera.h"
#include "BoundingBox.h"
#include "SceneManager.h"
#include "GPUTextureSampler.cuh"
#include "GPUTextures.h"

BOTH_CALLABLE float CUDAROP::GetMinElement(float val0, float val1, float val2)
{
	float min = val0;
	if (val1 < min)
		min = val1;
	if (val2 < min)
		min = val2;
	return min;
}

BOTH_CALLABLE float CUDAROP::GetMaxElement(float val0, float val1, float val2)
{
	float max = val0;
	if (val1 > max)
		max = val1;
	if (val2 > max)
		max = val2;
	return max;
}

GPU_CALLABLE OVertex CUDAROP::GetNDCVertex(const IVertex& vertex, const FPoint3& camPos,
	const FMatrix4& viewMatrix, const FMatrix4& projectionMatrix, const FMatrix4& worldMatrix)
{
	// combined worldViewProjectionMatrix
	const FMatrix4 worldViewProjectionMatrix = projectionMatrix * viewMatrix * worldMatrix;

	FPoint4 NDCspace = worldViewProjectionMatrix * FPoint4{ vertex.v.x, vertex.v.y, vertex.v.z, vertex.v.z };
	// converting to NDCspace
	NDCspace.x /= NDCspace.w;
	NDCspace.y /= NDCspace.w;
	NDCspace.z /= NDCspace.w;
	//NDCspace.w == NDCspace.w;

	// converting to raster-/screenspace
	//NDCspace.x = ((NDCspace.x + 1) / 2) * m_Width;
	//NDCspace.y = ((1 - NDCspace.y) / 2) * m_Height;
	// !DONE AFTER FRUSTUMTEST!

	// calculating vertex world position
	const FPoint3 worldPosition{ worldMatrix * FPoint4{ vertex.v } };
	const FVector3 viewDirection{ GetNormalized(worldPosition - camPos) };
	const FVector3 worldNormal{ worldMatrix * FVector4{ vertex.n } };
	const FVector3 worldTangent{ worldMatrix * FVector4{ vertex.tan } };

	return OVertex{ NDCspace, vertex.uv, worldNormal, worldTangent, vertex.c, viewDirection };
}

GPU_CALLABLE bool CUDAROP::EdgeFunction(const FVector2& v0, const FVector2& v1, const FPoint2& pixel)
{
	// counter-clockwise
	const FVector2 edge{ v0 - v1 };
	const FVector2 vertexToPixel{ pixel - v0 };
	const float cross = Cross(edge, vertexToPixel);
	//TODO: weight? (totalArea..)
	return cross < 0.f;
}

GPU_CALLABLE bool CUDAROP::IsPixelInTriangle(FPoint4 rasterCoords[3], const FPoint2& pixel, float weights[3])
{
	const FPoint2& v0 = rasterCoords[0].xy;
	const FPoint2& v1 = rasterCoords[1].xy;
	const FPoint2& v2 = rasterCoords[2].xy;

	const FVector2 edgeA{ v0 - v1 };
	const FVector2 edgeB{ v1 - v2 };
	const FVector2 edgeC{ v2 - v0 };
	// counter-clockwise

	bool isInTriangle{ true };
	const float totalArea = Cross(edgeA, edgeC);

	// edgeA
	FVector2 vertexToPixel{ pixel - v0 };
	float cross = Cross(edgeA, vertexToPixel);
	isInTriangle &= cross < 0.f;
	// weight2 == positive cross of 'previous' edge, for v2 this is edgeA (COUNTER-CLOCKWISE)
	weights[2] = cross / totalArea;

	// edgeB
	vertexToPixel = { pixel - v1 };
	cross = Cross(edgeB, vertexToPixel);
	isInTriangle &= cross < 0.f;
	// weight1 (for v1 this is edgeB)
	weights[1] = cross / totalArea;

	// edgeC
	vertexToPixel = { pixel - v2 };
	cross = Cross(edgeC, vertexToPixel);
	isInTriangle &= cross < 0.f;
	// weight0 (for v0 this is edgeC)
	weights[0] = cross / totalArea;

	//weights == inverted negative cross of 'previous' edge
	//weights[0] = Cross(-vertexToPixel, edgeC) / totalArea;
	//weights[1] = Cross(-vertexToPixel, edgeB) / totalArea;
	//weights[2] = Cross(-vertexToPixel, edgeA) / totalArea;
	// gives positive results because counter-clockwise
	//const float total = weights[0] + weights[1] + weights[2]; // total result equals 1

	//TODO use?
	//isInTriangle &= CUDAROP::EdgeFunction((FVector2)v0, (FVector2)v1, pixel); //edgeA
	//isInTriangle &= CUDAROP::EdgeFunction((FVector2)v1, (FVector2)v2, pixel); //edgeB
	//isInTriangle &= CUDAROP::EdgeFunction((FVector2)v2, (FVector2)v0, pixel); //edgeC

	return isInTriangle;
}

GPU_CALLABLE bool CUDAROP::DepthTest(FPoint4 rasterCoords[3], float& depthBuffer, float weights[3], float& zInterpolated)
{
	zInterpolated = (weights[0] * rasterCoords[0].z) + (weights[1] * rasterCoords[1].z) + (weights[2] * rasterCoords[2].z);

	if (zInterpolated < 0.f || zInterpolated > 1.f) return false;
	if (zInterpolated > depthBuffer) return false;

	depthBuffer = zInterpolated;

	return true;
}

GPU_CALLABLE bool CUDAROP::FrustumTestVertex(const OVertex& NDC)
{
	bool isPassed{ false };
	isPassed &= (NDC.v.x < -1.f || NDC.v.x > 1.f); // perspective divide X in NDC
	isPassed &= (NDC.v.y < -1.f || NDC.v.y > 1.f); // perspective divide Y in NDC
	isPassed &= (NDC.v.z < 0.f || NDC.v.z > 1.f); // perspective divide Z in NDC
	return isPassed;
}

GPU_CALLABLE bool CUDAROP::FrustumTest(const OVertex* NDC[3])
{
	bool isPassed{ false };
	isPassed &= CUDAROP::FrustumTestVertex(*NDC[0]);
	isPassed &= CUDAROP::FrustumTestVertex(*NDC[1]);
	isPassed &= CUDAROP::FrustumTestVertex(*NDC[2]);
	return isPassed;
}

GPU_CALLABLE void CUDAROP::NDCToScreenSpace(FPoint4 rasterCoords[3], const unsigned int width, const unsigned int height)
{
	for (int i{}; i < 3; ++i)
	{
		rasterCoords[i].x = ((rasterCoords[i].x + 1) / 2) * width;
		rasterCoords[i].y = ((1 - rasterCoords[i].y) / 2) * height;
	}
}

GPU_CALLABLE BoundingBox CUDAROP::GetBoundingBox(FPoint4 rasterCoords[3], const unsigned int width, const unsigned int height)
{
	BoundingBox bb;
	bb.xMin = (short)GetMinElement(rasterCoords[0].x, rasterCoords[1].x, rasterCoords[2].x) - 1; // xMin
	bb.yMin = (short)GetMinElement(rasterCoords[0].y, rasterCoords[1].y, rasterCoords[2].y) - 1; // yMin
	bb.xMax = (short)GetMaxElement(rasterCoords[0].x, rasterCoords[1].x, rasterCoords[2].x) + 1; // xMax
	bb.yMax = (short)GetMaxElement(rasterCoords[0].y, rasterCoords[1].y, rasterCoords[2].y) + 1; // yMax

	if (bb.xMin < 0) bb.xMin = 0; //clamp minX to Left of screen
	if (bb.yMin < 0) bb.yMin = 0; //clamp minY to Bottom of screen
	if (bb.xMax > width) bb.xMax = width; //clamp maxX to Right of screen
	if (bb.yMax > height) bb.yMax = height; //clamp maxY to Top of screen

	return bb;
}

GPU_CALLABLE void CUDAROP::RasterizeTriangle(OVertex* triangle[3], FPoint4 rasterCoords[3],
	unsigned int* pBufferPixels, float* pDepthBuffer, const unsigned int width, const unsigned int height)
{
	CUDAROP::NDCToScreenSpace(rasterCoords, width, height);
	const BoundingBox bb = CUDAROP::GetBoundingBox(rasterCoords, width, height);
	CUDAROP::RenderPixelsInTriangle(triangle, rasterCoords, pBufferPixels, pDepthBuffer, bb, width, height);
}

GPU_CALLABLE void CUDAROP::RenderPixelsInTriangle(OVertex* screenspaceTriangle[3], FPoint4 rasterCoords[3],
	unsigned int* pBufferPixels, float* pDepthBuffer, const BoundingBox& bb, const unsigned int width, const unsigned int height)
{
	const GPUTextures& textures{};
	SampleState sampleState{};
	bool isDepthColour{};

	const OVertex& v0 = *screenspaceTriangle[0];
	const OVertex& v1 = *screenspaceTriangle[1];
	const OVertex& v2 = *screenspaceTriangle[2];

	//Loop over all pixels in bounding box
	for (uint32_t r = bb.yMin; r < bb.yMax; ++r) // adding and subtracting 1 to get rid of seaming artifacts
	{
		for (uint32_t c = bb.xMin; c < bb.xMax; ++c)
		{
			const FPoint2 pixel{ float(c), float(r) };
			float weights[3];

			if (CUDAROP::IsPixelInTriangle(rasterCoords, pixel, weights))
			{
				float zInterpolated{};
				if (CUDAROP::DepthTest(rasterCoords, pDepthBuffer[size_t(c) + (size_t(r) * width)], weights, zInterpolated))
				{
					const float wInterpolated = (weights[0] * v0.v.w) + (weights[1] * v1.v.w) + (weights[2] * v2.v.w);

					FVector2 interpolatedUV{
						weights[0] * (v0.uv.x / rasterCoords[0].w) + weights[1] * (v1.uv.x / rasterCoords[1].w) + weights[2] * (v2.uv.x / rasterCoords[2].w),
						weights[0] * (v0.uv.y / rasterCoords[0].w) + weights[1] * (v1.uv.y / rasterCoords[1].w) + weights[2] * (v2.uv.y / rasterCoords[2].w) };
					interpolatedUV *= wInterpolated;

					FVector3 interpolatedNormal{
						 weights[0] * (v0.n.x / rasterCoords[0].w) + weights[1] * (v1.n.x / rasterCoords[1].w) + weights[2] * (v2.n.x / rasterCoords[2].w),
						 weights[0] * (v0.n.y / rasterCoords[0].w) + weights[1] * (v1.n.y / rasterCoords[1].w) + weights[2] * (v2.n.y / rasterCoords[2].w),
						 weights[0] * (v0.n.z / rasterCoords[0].w) + weights[1] * (v1.n.z / rasterCoords[1].w) + weights[2] * (v2.n.z / rasterCoords[2].w) };
					interpolatedNormal *= wInterpolated;

					const FVector3 interpolatedTangent{
						weights[0] * (v0.tan.x / rasterCoords[0].w) + weights[1] * (v1.tan.x / rasterCoords[1].w) + weights[2] * (v2.tan.x / rasterCoords[2].w),
						weights[0] * (v0.tan.y / rasterCoords[0].w) + weights[1] * (v1.tan.y / rasterCoords[1].w) + weights[2] * (v2.tan.y / rasterCoords[2].w),
						weights[0] * (v0.tan.z / rasterCoords[0].w) + weights[1] * (v1.tan.z / rasterCoords[1].w) + weights[2] * (v2.tan.z / rasterCoords[2].w) };

					FVector3 interpolatedViewDirection{
					weights[0] * (v0.vd.y / rasterCoords[0].w) + weights[1] * (v1.vd.y / rasterCoords[1].w) + weights[2] * (v2.vd.y / rasterCoords[2].w),
					weights[0] * (v0.vd.x / rasterCoords[0].w) + weights[1] * (v1.vd.x / rasterCoords[1].w) + weights[2] * (v2.vd.x / rasterCoords[2].w),
					weights[0] * (v0.vd.z / rasterCoords[0].w) + weights[1] * (v1.vd.z / rasterCoords[1].w) + weights[2] * (v2.vd.z / rasterCoords[2].w) };
					Normalize(interpolatedViewDirection);

					const RGBColor interpolatedColour{
						weights[0] * (v0.c.r / rasterCoords[0].w) + weights[1] * (v1.c.r / rasterCoords[1].w) + weights[2] * (v2.c.r / rasterCoords[2].w),
						weights[0] * (v0.c.g / rasterCoords[0].w) + weights[1] * (v1.c.g / rasterCoords[1].w) + weights[2] * (v2.c.g / rasterCoords[2].w),
						weights[0] * (v0.c.b / rasterCoords[0].w) + weights[1] * (v1.c.b / rasterCoords[1].w) + weights[2] * (v2.c.b / rasterCoords[2].w) };

					OVertex oVertex;
					oVertex.v = FPoint4{ pixel, zInterpolated, wInterpolated };
					oVertex.c = std::move(interpolatedColour);
					oVertex.uv = std::move(interpolatedUV);
					oVertex.n = std::move(interpolatedNormal);
					oVertex.tan = std::move(interpolatedTangent);
					oVertex.vd = std::move(interpolatedViewDirection);

					const RGBColor finalColour = CUDAROP::ShadePixel(oVertex, textures, sampleState, isDepthColour);

					// final draw
					unsigned char* pRGBA = (unsigned char*)pBufferPixels[(int)oVertex.v.x + (int)(oVertex.v.y * width)];
					pRGBA[0] = static_cast<uint8_t>(finalColour.r * 255.f);
					pRGBA[1] = static_cast<uint8_t>(finalColour.g * 255.f);
					pRGBA[2] = static_cast<uint8_t>(finalColour.b * 255.f);
					//pRGBA[3] = static_cast<uint8_t>(finalColour.a * 255.f);
					//SDL_MapRGB(pBuffer->format, static_cast<uint8_t>(finalColour.r * 255.f), static_cast<uint8_t>(finalColour.g * 255.f), static_cast<uint8_t>(finalColour.b * 255.f));
				}
			}
		}
	}
}

GPU_CALLABLE RGBColor CUDAROP::ShadePixel(const OVertex& oVertex, const GPUTextures& textures, SampleState sampleState, bool isDepthColour)
{
	RGBColor finalColour{};
	if (isDepthColour)
	{
		finalColour = RGBColor{ Remap(oVertex.v.z, 0.985f, 1.f), 0.f, 0.f }; // depth colour
		finalColour.ClampColor();
	}
	else
	{
		const RGBColor diffuseColour = GPUTextureSampler::Sample(textures.pDiff, oVertex.uv, sampleState);

		const RGBColor normalRGB = GPUTextureSampler::Sample(textures.pNorm, oVertex.uv, sampleState);
		FVector3 normal{ normalRGB.r, normalRGB.g, normalRGB.b };

		FVector3 binormal{ Cross(oVertex.tan, oVertex.n) };
		FMatrix3 tangentSpaceAxis{ oVertex.tan, binormal, oVertex.n };

		normal.x = 2.f * normal.x - 1.f;
		normal.y = 2.f * normal.y - 1.f;
		normal.z = 2.f * normal.z - 1.f;

		normal = tangentSpaceAxis * normal;

		//// light calculations
		//for (Light* pLight : sm.GetSceneGraph()->GetLights())
		//{
		//	const FVector3& lightDir{ pLight->GetDirection(FPoint3{}) };
		//	const float observedArea{ Dot(-normal, lightDir) };
		//
		//	if (observedArea < 0.f)
		//		continue;
		//
		//	const RGBColor biradiance{ pLight->GetBiradiance(FPoint3{}) };
		//	// swapped direction of lights
		//
		//	// phong
		//	const FVector3 reflectV{ Reflect(lightDir, normal) };
		//	//Normalize(reflectV);
		//	const float angle{ Dot(reflectV, oVertex.vd) };
		//	const RGBColor specularSample{ textures.pSpec->Sample(oVertex.uv, sampleState) };
		//	const RGBColor phongSpecularReflection{ specularSample * powf(angle, textures.pGloss->Sample(oVertex.uv, sampleState).r * 25.f) };
		//
		//	//const RGBColor lambertColour{ diffuseColour * (RGBColor{1.f,1.f,1.f} - specularSample) };
		//	//const RGBColor lambertColour{ (diffuseColour / float(E_PI)) * (RGBColor{1.f,1.f,1.f} - specularSample) };
		//	const RGBColor lambertColour{ (diffuseColour * specularSample) / float(E_PI) }; //severely incorrect result, using diffusecolour for now
		//	// Lambert diffuse == incoming colour multiplied by diffuse coefficient (1 in this case) divided by Pi
		//	finalColour += biradiance * (diffuseColour + phongSpecularReflection) * observedArea;
		//}
		finalColour.ClampColor();
	}
	return finalColour;
}