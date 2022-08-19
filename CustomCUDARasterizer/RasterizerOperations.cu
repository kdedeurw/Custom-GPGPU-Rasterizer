#include "PCH.h"

//Project CUDA includes
#include "GPUTextureSampler.cuh"

BOTH_CALLABLE static
float GetMinElement(float val0, float val1, float val2)
{
	float min = val0;
	if (val1 < min)
		min = val1;
	if (val2 < min)
		min = val2;
	return min;
}

BOTH_CALLABLE static
float GetMaxElement(float val0, float val1, float val2)
{
	float max = val0;
	if (val1 > max)
		max = val1;
	if (val2 > max)
		max = val2;
	return max;
}

GPU_CALLABLE GPU_INLINE static
float EdgeFunction(const FPoint2& v, const FVector2& edge, const FPoint2& pixel)
{
	// clockwise
	const FVector2 vertexToPixel{ pixel - v };
	return Cross(vertexToPixel, edge);
}

GPU_CALLABLE static
bool IsPixelInTriangle(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2, const FPoint2& pixel, float weights[3])
{
	const FVector2 edgeA{ v1.xy - v0.xy };
	const FVector2 edgeB{ v2.xy - v1.xy };
	const FVector2 edgeC{ v0.xy - v2.xy };
	// clockwise
	//const FVector2 edgeA{ v0.xy - v1.xy };
	//const FVector2 edgeB{ v1.xy - v2.xy };
	//const FVector2 edgeC{ v2.xy - v0.xy };
	// counter-clockwise

	weights[2] = EdgeFunction(v0.xy, edgeA, pixel);
	weights[0] = EdgeFunction(v1.xy, edgeB, pixel);
	weights[1] = EdgeFunction(v2.xy, edgeC, pixel);

	return weights[0] >= 0.f && weights[1] >= 0.f && weights[2] >= 0.f;
}

GPU_CALLABLE static
bool IsAllXOutsideFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return	(v0.x < -1.f && v1.x < -1.f && v2.x < -1.f) ||
		(v0.x > 1.f && v1.x > 1.f && v2.x > 1.f);
}

GPU_CALLABLE static
bool IsAllYOutsideFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return	(v0.y < -1.f && v1.y < -1.f && v2.y < -1.f) ||
		(v0.y > 1.f && v1.y > 1.f && v2.y > 1.f);
}

GPU_CALLABLE static
bool IsAllZOutsideFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return	(v0.z < 0.f && v1.z < 0.f && v2.z < 0.f) ||
		(v0.z > 1.f && v1.z > 1.f && v2.z > 1.f);
}

GPU_CALLABLE static
bool IsTriangleVisible(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	// Solution to FrustumCulling bug
	//	   if (all x values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all y values are < -1.f or > 1.f) AT ONCE, cull
	//	|| if (all z values are < 0.f or > 1.f) AT ONCE, cull
	return(!IsAllXOutsideFrustum(v0, v1, v2)
		&& !IsAllYOutsideFrustum(v0, v1, v2)
		&& !IsAllZOutsideFrustum(v0, v1, v2));
}

GPU_CALLABLE static
bool IsVertexInFrustum(const FPoint4& NDC)
{
	return!((NDC.x < -1.f || NDC.x > 1.f) ||
		(NDC.y < -1.f || NDC.y > 1.f) ||
		(NDC.z < 0.f || NDC.z > 1.f));
}

GPU_CALLABLE static
bool IsTriangleInFrustum(const FPoint4& v0, const FPoint4& v1, const FPoint4& v2)
{
	return(IsVertexInFrustum(v0)
		|| IsVertexInFrustum(v1)
		|| IsVertexInFrustum(v2));
	//TODO: bug, triangles gets culled when zoomed in, aka all 3 vertices are outside of frustum
}

BOTH_CALLABLE static
BoundingBox GetBoundingBox(const FPoint2& v0, const FPoint2& v1, const FPoint2& v2, const unsigned int width, const unsigned int height)
{
	BoundingBox bb;
	bb.xMin = (short)GetMinElement(v0.x, v1.x, v2.x) - 1; // xMin
	bb.yMin = (short)GetMinElement(v0.y, v1.y, v2.y) - 1; // yMin
	bb.xMax = (short)GetMaxElement(v0.x, v1.x, v2.x) + 1; // xMax
	bb.yMax = (short)GetMaxElement(v0.y, v1.y, v2.y) + 1; // yMax

	if (bb.xMin < 0) bb.xMin = 0; //clamp minX to Left of screen
	if (bb.yMin < 0) bb.yMin = 0; //clamp minY to Bottom of screen
	if (bb.xMax > width) bb.xMax = width; //clamp maxX to Right of screen
	if (bb.yMax > height) bb.yMax = height; //clamp maxY to Top of screen

	return bb;
}

BOTH_CALLABLE GPU_INLINE static
void NDCToScreenSpace(FPoint2& v0, FPoint2& v1, FPoint2& v2, const unsigned int width, const unsigned int height)
{
	v0.x = ((v0.x + 1) / 2) * width;
	v0.y = ((1 - v0.y) / 2) * height;
	v1.x = ((v1.x + 1) / 2) * width;
	v1.y = ((1 - v1.y) / 2) * height;
	v2.x = ((v2.x + 1) / 2) * width;
	v2.y = ((1 - v2.y) / 2) * height;
}

GPU_CALLABLE GPU_INLINE static
OVertex GetNDCVertex(const IVertex& __restrict__ iVertex, const FMatrix4& wvpMat, const FMatrix4& worldMat, const FMatrix3& rotMat, const FPoint3& camPos)
{
	OVertex oVertex;
	oVertex.p = wvpMat * FPoint4{ iVertex.p };
	oVertex.p.x /= oVertex.p.w;
	oVertex.p.y /= oVertex.p.w;
	oVertex.p.z /= oVertex.p.w;

	oVertex.n = rotMat * iVertex.n;
	oVertex.tan = rotMat * iVertex.tan;

	const FPoint3 worldPosition{ worldMat * FPoint4{ iVertex.p } };
	oVertex.vd = GetNormalized(worldPosition - camPos);

	oVertex.uv = iVertex.uv;
	oVertex.c = iVertex.c;

	return oVertex;
}

#pragma region DEPRECATED

GPU_CALLABLE static
bool IsPixelInTriangle(const RasterTriangle& triangle, const FPoint2& pixel, float weights[3])
{
	return IsPixelInTriangle(triangle.v0, triangle.v1, triangle.v2, pixel, weights);
}

GPU_CALLABLE static
bool IsAllXOutsideFrustum(const RasterTriangle& triangle)
{
	return IsAllXOutsideFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsAllYOutsideFrustum(const RasterTriangle& triangle)
{
	return IsAllYOutsideFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsAllZOutsideFrustum(const RasterTriangle& triangle)
{
	return IsAllZOutsideFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsTriangleVisible(const RasterTriangle& triangle)
{
	return IsTriangleVisible(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
bool IsTriangleInFrustum(const RasterTriangle& triangle)
{
	return IsTriangleInFrustum(triangle.v0, triangle.v1, triangle.v2);
}

GPU_CALLABLE static
void NDCToScreenSpace(RasterTriangle& triangle, const unsigned int width, const unsigned int height)
{
	NDCToScreenSpace(triangle.v0.xy, triangle.v1.xy, triangle.v2.xy, width, height);
}

GPU_CALLABLE static
BoundingBox GetBoundingBox(const RasterTriangle& triangle, const unsigned int width, const unsigned int height)
{
	return GetBoundingBox(triangle.v0.xy, triangle.v1.xy, triangle.v2.xy, width, height);
}

#pragma endregion

GPU_CALLABLE static
bool IsPixelInBoundingBox(const FPoint2& pixel, const BoundingBox& bb)
{
	return pixel.x < bb.xMin || pixel.x > bb.xMax || pixel.y < bb.yMin || pixel.y > bb.yMax;
}

GPU_CALLABLE GPU_INLINE static
unsigned int GetStridedIdxByOffset(unsigned int globalDataIdx, unsigned int vertexStride, unsigned int valueStride, unsigned int offset = 0)
{
	//what value in row of [0, valueStride] + what vertex globally + element offset
	return (threadIdx.x % valueStride) + (globalDataIdx / valueStride) * vertexStride + offset;
}

GPU_CALLABLE static
void PerformDepthTestAtomic(int* dev_DepthBuffer, int* dev_DepthMutexBuffer, const unsigned int pixelIdx, float zInterpolated, PixelShade* dev_PixelShadeBuffer, const PixelShade& pixelShade)
{
	//Update depthbuffer atomically
	bool isDone = false;
	do
	{
		isDone = (atomicCAS(&dev_DepthMutexBuffer[pixelIdx], 0, 1) == 0);
		if (isDone)
		{
			//critical section
			if (zInterpolated < dev_DepthBuffer[pixelIdx])
			{
				dev_DepthBuffer[pixelIdx] = zInterpolated;
				dev_PixelShadeBuffer[pixelIdx] = pixelShade;
			}
			dev_DepthMutexBuffer[pixelIdx] = 0;
			//end of critical section
		}
	} while (!isDone);
}

GPU_CALLABLE static
void RasterizePixel(const FPoint2& pixel, const OVertex& v0, const OVertex& v1, const OVertex& v2,
	int* dev_DepthBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const GPUTexturesCompact& textures)
{
	const float v0InvDepth = 1.f / v0.p.w;
	const float v1InvDepth = 1.f / v1.p.w;
	const float v2InvDepth = 1.f / v2.p.w;

	float weights[3];
	if (IsPixelInTriangle(v0.p, v1.p, v2.p, pixel, weights))
	{
		const float totalArea = abs(Cross(v0.p.xy - v1.p.xy, v0.p.xy - v2.p.xy));
		weights[0] /= totalArea;
		weights[1] /= totalArea;
		weights[2] /= totalArea;

		const float zInterpolated = (weights[0] * v0.p.z) + (weights[1] * v1.p.z) + (weights[2] * v2.p.z);

		//peform early depth test
		if (zInterpolated < 0 || zInterpolated > 1.f)
			return;

		const float wInterpolated = 1.f / (v0InvDepth * weights[0] + v1InvDepth * weights[1] + v2InvDepth * weights[2]);

		//create pixelshade object (== fragment)
		PixelShade pixelShade;

		//depthbuffer visualisation
		pixelShade.zInterpolated = zInterpolated;
		pixelShade.wInterpolated = wInterpolated;

		//uv
		pixelShade.uv.x = weights[0] * (v0.uv.x * v0InvDepth) + weights[1] * (v1.uv.x * v1InvDepth) + weights[2] * (v2.uv.x * v2InvDepth);
		pixelShade.uv.y = weights[0] * (v0.uv.y * v0InvDepth) + weights[1] * (v1.uv.y * v1InvDepth) + weights[2] * (v2.uv.y * v2InvDepth);
		pixelShade.uv *= wInterpolated;

		//normal
		pixelShade.n.x = weights[0] * (v0.n.x * v0InvDepth) + weights[1] * (v1.n.x * v1InvDepth) + weights[2] * (v2.n.x * v2InvDepth);
		pixelShade.n.y = weights[0] * (v0.n.y * v0InvDepth) + weights[1] * (v1.n.y * v1InvDepth) + weights[2] * (v2.n.y * v2InvDepth);
		pixelShade.n.z = weights[0] * (v0.n.z * v0InvDepth) + weights[1] * (v1.n.z * v1InvDepth) + weights[2] * (v2.n.z * v2InvDepth);
		pixelShade.n *= wInterpolated;

		//tangent
		pixelShade.tan.x = weights[0] * (v0.tan.x * v0InvDepth) + weights[1] * (v1.tan.x * v1InvDepth) + weights[2] * (v2.tan.x * v2InvDepth);
		pixelShade.tan.y = weights[0] * (v0.tan.y * v0InvDepth) + weights[1] * (v1.tan.y * v1InvDepth) + weights[2] * (v2.tan.y * v2InvDepth);
		pixelShade.tan.z = weights[0] * (v0.tan.z * v0InvDepth) + weights[1] * (v1.tan.z * v1InvDepth) + weights[2] * (v2.tan.z * v2InvDepth);

		//view direction
		pixelShade.vd.x = weights[0] * (v0.vd.x * v0InvDepth) + weights[1] * (v1.vd.x * v1InvDepth) + weights[2] * (v2.vd.x * v2InvDepth);
		pixelShade.vd.y = weights[0] * (v0.vd.y * v0InvDepth) + weights[1] * (v1.vd.y * v1InvDepth) + weights[2] * (v2.vd.y * v2InvDepth);
		pixelShade.vd.z = weights[0] * (v0.vd.z * v0InvDepth) + weights[1] * (v1.vd.z * v1InvDepth) + weights[2] * (v2.vd.z * v2InvDepth);
		Normalize(pixelShade.vd);

		//colour
		const RGBColor interpolatedColour{
			weights[0] * v0.c.r + weights[1] * v1.c.r + weights[2] * v2.c.r,
			weights[0] * v0.c.g + weights[1] * v1.c.g + weights[2] * v2.c.g,
			weights[0] * v0.c.b + weights[1] * v1.c.b + weights[2] * v2.c.b };
		pixelShade.colour = RGBA::GetRGBAFromColour(interpolatedColour).colour32;

		//store textures
		pixelShade.textures = textures;

		//multiplying z value by a INT_MAX because atomicCAS only accepts ints
		const int scaledZ = zInterpolated * INT_MAX;

		const unsigned int pixelIdx = (unsigned int)pixel.x + (unsigned int)pixel.y * width;

		if (scaledZ < dev_DepthBuffer[pixelIdx])
		{
			dev_DepthBuffer[pixelIdx] = scaledZ;
			dev_PixelShadeBuffer[pixelIdx] = pixelShade;
		}
	}
}

GPU_CALLABLE static
void RasterizePixelAtomic(const FPoint2& pixel, const OVertex& v0, const OVertex& v1, const OVertex& v2,
	int* dev_DepthBuffer, int* dev_DepthMutexBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const GPUTexturesCompact& textures)
{
	const float v0InvDepth = 1.f / v0.p.w;
	const float v1InvDepth = 1.f / v1.p.w;
	const float v2InvDepth = 1.f / v2.p.w;

	float weights[3];
	if (IsPixelInTriangle(v0.p, v1.p, v2.p, pixel, weights))
	{
		const float totalArea = abs(Cross(v0.p.xy - v1.p.xy, v0.p.xy - v2.p.xy));
		weights[0] /= totalArea;
		weights[1] /= totalArea;
		weights[2] /= totalArea;

		const float zInterpolated = (weights[0] * v0.p.z) + (weights[1] * v1.p.z) + (weights[2] * v2.p.z);

		//peform early depth test
		if (zInterpolated < 0 || zInterpolated > 1.f)
			return;

		const float wInterpolated = 1.f / (v0InvDepth * weights[0] + v1InvDepth * weights[1] + v2InvDepth * weights[2]);

		//create pixelshade object (== fragment)
		PixelShade pixelShade;

		//depthbuffer visualisation
		pixelShade.zInterpolated = zInterpolated;
		pixelShade.wInterpolated = wInterpolated;

		//uv
		pixelShade.uv.x = weights[0] * (v0.uv.x * v0InvDepth) + weights[1] * (v1.uv.x * v1InvDepth) + weights[2] * (v2.uv.x * v2InvDepth);
		pixelShade.uv.y = weights[0] * (v0.uv.y * v0InvDepth) + weights[1] * (v1.uv.y * v1InvDepth) + weights[2] * (v2.uv.y * v2InvDepth);
		pixelShade.uv *= wInterpolated;

		//normal
		pixelShade.n.x = weights[0] * (v0.n.x * v0InvDepth) + weights[1] * (v1.n.x * v1InvDepth) + weights[2] * (v2.n.x * v2InvDepth);
		pixelShade.n.y = weights[0] * (v0.n.y * v0InvDepth) + weights[1] * (v1.n.y * v1InvDepth) + weights[2] * (v2.n.y * v2InvDepth);
		pixelShade.n.z = weights[0] * (v0.n.z * v0InvDepth) + weights[1] * (v1.n.z * v1InvDepth) + weights[2] * (v2.n.z * v2InvDepth);
		pixelShade.n *= wInterpolated;

		//tangent
		pixelShade.tan.x = weights[0] * (v0.tan.x * v0InvDepth) + weights[1] * (v1.tan.x * v1InvDepth) + weights[2] * (v2.tan.x * v2InvDepth);
		pixelShade.tan.y = weights[0] * (v0.tan.y * v0InvDepth) + weights[1] * (v1.tan.y * v1InvDepth) + weights[2] * (v2.tan.y * v2InvDepth);
		pixelShade.tan.z = weights[0] * (v0.tan.z * v0InvDepth) + weights[1] * (v1.tan.z * v1InvDepth) + weights[2] * (v2.tan.z * v2InvDepth);

		//view direction
		pixelShade.vd.x = weights[0] * (v0.vd.x * v0InvDepth) + weights[1] * (v1.vd.x * v1InvDepth) + weights[2] * (v2.vd.x * v2InvDepth);
		pixelShade.vd.y = weights[0] * (v0.vd.y * v0InvDepth) + weights[1] * (v1.vd.y * v1InvDepth) + weights[2] * (v2.vd.y * v2InvDepth);
		pixelShade.vd.z = weights[0] * (v0.vd.z * v0InvDepth) + weights[1] * (v1.vd.z * v1InvDepth) + weights[2] * (v2.vd.z * v2InvDepth);
		Normalize(pixelShade.vd);

		//colour
		const RGBColor interpolatedColour{
			weights[0] * v0.c.r + weights[1] * v1.c.r + weights[2] * v2.c.r,
			weights[0] * v0.c.g + weights[1] * v1.c.g + weights[2] * v2.c.g,
			weights[0] * v0.c.b + weights[1] * v1.c.b + weights[2] * v2.c.b };
		pixelShade.colour = RGBA::GetRGBAFromColour(interpolatedColour).colour32;

		//store textures
		pixelShade.textures = textures;

		//multiplying z value by a INT_MAX because atomicCAS only accepts ints
		const int scaledZ = zInterpolated * INT_MAX;

		const unsigned int pixelIdx = (unsigned int)pixel.x + (unsigned int)pixel.y * width;

		PerformDepthTestAtomic(dev_DepthBuffer, dev_DepthMutexBuffer, pixelIdx, scaledZ, dev_PixelShadeBuffer, pixelShade);
	}
}

GPU_CALLABLE static
void RasterizeTriangle(const BoundingBox& bb, const OVertex& v0, const OVertex& v1, const OVertex& v2,
	int* dev_DepthMutexBuffer, int* dev_DepthBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const GPUTexturesCompact& textures)
{
	//Loop over all pixels in bounding box
	for (unsigned short y = bb.yMin; y < bb.yMax; ++y)
	{
		for (unsigned short x = bb.xMin; x < bb.xMax; ++x)
		{
			const FPoint2 pixel{ float(x), float(y) };
			RasterizePixelAtomic(pixel, v0, v1, v2, dev_DepthBuffer, dev_DepthMutexBuffer, dev_PixelShadeBuffer, width, textures);
		}
	}
}

GPU_CALLABLE GPU_INLINE static
RGBColor ShadePixel(const GPUTexturesCompact& textures, const FVector2& uv, const FVector3& n, const FVector3& tan, const FVector3& vd,
	SampleState sampleState, bool isFlipGreenChannel = false)
{
	RGBColor finalColour{};

	//global settings
	const RGBColor ambientColour{ 0.05f, 0.05f, 0.05f };
	const FVector3 lightDirection = { 0.577f, -0.577f, -0.577f };
	const float lightIntensity = 7.0f;

	// texture sampling
	const RGBColor diffuseSample = GPUTextureSampler::Sample(textures.Diff, textures.w, textures.h, uv, sampleState);

	if (textures.Norm.dev_pTex != 0)
	{
		const RGBColor normalSample = GPUTextureSampler::Sample(textures.Norm, textures.w, textures.h, uv, sampleState);

		// normal mapping
		FVector3 binormal = Cross(tan, n);
		if (isFlipGreenChannel)
			binormal = -binormal;
		const FMatrix3 tangentSpaceAxis{ tan, binormal, n };

		FVector3 finalNormal{ 2.f * normalSample.r - 1.f, 2.f * normalSample.g - 1.f, 2.f * normalSample.b - 1.f };
		finalNormal = tangentSpaceAxis * finalNormal;

		// light calculations
		float observedArea{ Dot(-finalNormal, lightDirection) };
		Clamp(observedArea, 0.f, observedArea);
		observedArea /= (float)PI;
		observedArea *= lightIntensity;
		const RGBColor diffuseColour = diffuseSample * observedArea;

		if (textures.Spec.dev_pTex != 0 && textures.Gloss.dev_pTex != 0)
		{
			const RGBColor specularSample = GPUTextureSampler::Sample(textures.Spec, textures.w, textures.h, uv, sampleState);
			const RGBColor glossSample = GPUTextureSampler::Sample(textures.Gloss, textures.w, textures.h, uv, sampleState);

			// phong specular
			const FVector3 reflectV{ Reflect(lightDirection, finalNormal) };
			float angle{ Dot(reflectV, vd) };
			Clamp(angle, 0.f, 1.f);
			const float shininess = 25.f;
			angle = powf(angle, glossSample.r * shininess);
			const RGBColor specularColour = specularSample * angle;

			// final
			finalColour = ambientColour + diffuseColour + specularColour;
			finalColour.ClampColor();
		}
		else
		{
			finalColour = diffuseColour;
		}
	}
	else
	{
		finalColour = diffuseSample;
	}
	return finalColour;
}

GPU_CALLABLE GPU_INLINE static
void MultiplyMatVec(const float* pMat, float* pVec, unsigned int matSize, unsigned int vecSize)
{
	//thread goes through each element of vector
	float vec[4]{};
	for (unsigned int element{}; element < vecSize; ++element)
	{
		float sum{};
		for (unsigned int i{}; i < matSize; ++i)
		{
			sum += pMat[(element * matSize) + i] * pVec[i];
		}
		vec[element] = sum;
	}
	memcpy(pVec, vec, vecSize * 4);
}

GPU_CALLABLE GPU_INLINE static
void CalculateOutputPosXYZ(const float* pMat, float* pVec)
{
	constexpr unsigned int matSize = 4;
	constexpr unsigned int vecSize = 3;

	//thread goes through each element of vector
	float vec[3]{};
	for (unsigned int element{}; element < vecSize; ++element)
	{
		for (unsigned int i{}; i < vecSize; ++i)
		{
			vec[element] += pMat[(element * matSize) + i] * pVec[i];
		}
		vec[element] += pMat[(element * matSize) + 3]; // * pVec[w] == 1.f
	}
	memcpy(pVec, vec, 12);
}

GPU_CALLABLE GPU_INLINE static
void CalculateOutputPosXYZW(const float* pMat, float* pVec, float* pW)
{
	constexpr unsigned int matSize = 4;
	constexpr unsigned int vecSize = 3;

	//thread goes through each element of vector
	float vec[4]{};
	for (unsigned int element{}; element < vecSize; ++element)
	{
		for (unsigned int i{}; i < vecSize; ++i)
		{
			vec[element] += pMat[(element * matSize) + i] * pVec[i];
		}
		vec[element] += pMat[(element * matSize) + 3]; // * pVec[w] == 1.f
	}

	for (unsigned int i{}; i < vecSize; ++i)
	{
		vec[3] += pMat[12 + i] * pVec[i];
	}
	vec[3] += pMat[15]; // * pVec[w] == 1.f

	memcpy(pVec, vec, 12);
	*pW = vec[3];
}

//BINNING + TILING

GPU_CALLABLE static
BoundingBox GetBoundingBoxTiled(const FPoint2& v0, const FPoint2& v1, const FPoint2& v2,
	const unsigned int minX, const unsigned int minY, const unsigned int maxX, const unsigned int maxY)
{
	BoundingBox bb;
	bb.xMin = (short)GetMinElement(v0.x, v1.x, v2.x) - 1; // xMin
	bb.yMin = (short)GetMinElement(v0.y, v1.y, v2.y) - 1; // yMin
	bb.xMax = (short)GetMaxElement(v0.x, v1.x, v2.x) + 1; // xMax
	bb.yMax = (short)GetMaxElement(v0.y, v1.y, v2.y) + 1; // yMax

	if (bb.xMin < minX) bb.xMin = minX; //clamp minX to Left of screen
	if (bb.yMin < minY) bb.yMin = minY; //clamp minY to Bottom of screen
	if (bb.xMax > maxX) bb.xMax = maxX; //clamp maxX to Right of screen
	if (bb.yMax > maxY) bb.yMax = maxY; //clamp maxY to Top of screen

	return bb;
}

GPU_CALLABLE static
bool IsOverlapping(const BoundingBox& rect0, const BoundingBox& rect1)
{
	//if x is either left or right outside of other
	if (rect0.xMin > rect1.xMax || rect0.xMax < rect1.xMin)
		return false;

	//if y is either top or bottom outside of other
	if (rect0.yMin > rect1.yMax || rect0.yMax < rect1.yMin)
		return false;

	return true;
}