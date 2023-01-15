#pragma once
#include "GPUHelpers.h"

#include "Math.h"
#include "RGBColor.h"
#include "CUDATexture.h"
#include "RGBRaw.h"
#include "SampleState.h"
//#include "CUDAROPs.cu"
#include "CUDAStructs.h"

namespace CUDATextureSampler
{
	GPU_CALLABLE GPU_INLINE RGBColor Sample1D(const cudaTextureObject_t texObj, unsigned int w, unsigned int h, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor Sample2D(const cudaTextureObject_t texObj, unsigned int w, unsigned int h, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor Sample(const CUDATextureCompact& texture, const FVector2& uv, SampleState sampleState);
	GPU_CALLABLE GPU_INLINE RGBColor Sample(const unsigned int* pixels, unsigned int w, unsigned int h, const FVector2& uv, SampleState sampleState);
	GPU_CALLABLE GPU_INLINE RGBColor SamplePoint(const CUDATextureCompact& texture, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor SamplePoint(const unsigned int* pixels, unsigned int w, unsigned int h, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor SampleLinear(const CUDATextureCompact& texture, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor SampleLinear(const unsigned int* pixels, unsigned int w, unsigned int h, const FVector2& uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::Sample(const CUDATextureCompact& texture, const FVector2& uv, SampleState sampleState)
{
	if (sampleState == SampleState::Linear)
		return CUDATextureSampler::SampleLinear(texture, uv);

	return CUDATextureSampler::SamplePoint(texture, uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::Sample(const unsigned int* pixels, unsigned int w, unsigned int h, const FVector2& uv, SampleState sampleState)
{
	if (sampleState == SampleState::Linear)
		return CUDATextureSampler::SampleLinear(pixels, w, h, uv);

	return CUDATextureSampler::SamplePoint(pixels, w, h, uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::Sample1D(const cudaTextureObject_t texObj, unsigned int w, unsigned int h, const FVector2& uv)
{
	//// Transform coordinates
	float u{ uv.x };
	float v{ uv.y };
	u *= w;
	v *= h;
	//u += 0.5f;
	//v += 0.5f;
	//u -= 0.5f;
	//v -= 0.5f;
	//float tu = u * cosf(theta) – v * sinf(theta) + 0.5f;
	//float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

	unsigned int sample = tex1Dfetch<unsigned int>(texObj, u + v * w);
	const RGBA rgba{ sample };
	return RGBColor{ rgba.b8 / 255.f, rgba.g8 / 255.f, rgba.r8 / 255.f };
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::Sample2D(const cudaTextureObject_t texObj, unsigned int w, unsigned int h, const FVector2& uv)
{
	//// Transform coordinates
	float u{ uv.x };
	float v{ uv.y };
	//u += 0.5f;
	//v += 0.5f;
	//u -= 0.5f;
	//v -= 0.5f;
	//float tu = u * cosf(theta) – v * sinf(theta) + 0.5f;
	//float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

	unsigned int sample = tex2D<unsigned int>(texObj, u, v);
	const RGBA rgba{ sample };
	return RGBColor{ rgba.b8 / 255.f, rgba.g8 / 255.f, rgba.r8 / 255.f };
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::SamplePoint(const CUDATextureCompact& texture, const FVector2& uv)
{
	return SamplePoint(texture.dev_TextureData, texture.w, texture.h, uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::SamplePoint(const unsigned int* pixels, unsigned int w, unsigned int h, const FVector2& uv)
{
	// Transform coordinates
	const int u = int(uv.x * w + 0.5f);
	const int v = int(uv.y * h + 0.5f);

	//TODO: FastClamp instead of if statement
	//int newU = ClampFast(u, 0, int(w - 1));
	//int newV = ClampFast(v, 0, int(h - 1));

	if (u < 0 || v < 0 || u >= w || v >= h)
		return RGBColor{ 1.f, 0.f, 1.f };

	const int sampleIdx = u + v * w;
	unsigned int sample = pixels[sampleIdx];
	RGBColor sampleColour = RGBA::GetRGBColor(sample);
	return sampleColour;
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::SampleLinear(const CUDATextureCompact& texture, const FVector2& uv)
{
	return SampleLinear(texture.dev_TextureData, texture.w, texture.h, uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor CUDATextureSampler::SampleLinear(const unsigned int* pixels, unsigned int w, unsigned int h, const FVector2& uv)
{
	//Step 1: find pixel to sample from
	const int x = int(uv.x * w + 0.5f);
	const int y = int(uv.y * h + 0.5f);

	if (x < 0 || y < 0 || x >= w || y >= h)
		return RGBColor{ 1.f, 0.f, 1.f };

	//Step 2: find 4 neighbours
	const int texSize = w * h;
	const int originalPixel = x + y * w;

	const unsigned short numNeighbours = 4;
	int neighbours[numNeighbours];
	//sample 4 adjacent neighbours
	neighbours[0] = originalPixel - 1; //original pixel - 1x
	if (neighbours[0] < 0)
		neighbours[0] = originalPixel;
	//possible issue: x might shove back from left to right - 1y
	neighbours[1] = originalPixel + 1; //original pixel + 1x
	if (neighbours[1] >= texSize)
		neighbours[1] = originalPixel;
	//possible issue: x might shove back from right to left + 1y
	neighbours[2] = originalPixel + w; //original pixel + 1y
	if (neighbours[2] >= texSize)
		neighbours[2] = originalPixel;
	neighbours[3] = originalPixel - w; //original pixel - 1y
	if (neighbours[3] < 0)
		neighbours[3] = originalPixel;

	//get other 4 corner neighbours
	//pixels[4] = float(x - 1 + ((y - 1) * m_pSurface->w)); //original pixel - 1x -1y
	//if (pixels[4] < 0)
	//	pixels[4] = 0;
	//pixels[5] = float(x + 1 + ((y + 1) * m_pSurface->w)); //original pixel + 1x + 1y
	//if (pixels[5] < max)
	//	pixels[5] = max;
	//pixels[6] = float(x - 1 + ((y + 1) * m_pSurface->w)); //original pixel - 1x + 1y
	//if (pixels[6] < max)
	//	pixels[6] = max;
	//pixels[7] = float(x + 1 + ((y - 1) * m_pSurface->w)); //original pixel + 1x - 1y
	//if (pixels[7] < 0)
	//	pixels[7] = 0;

	//Step 3: define weights
	const float weight = 1.f / (numNeighbours * 2); //# pixels, equally divided
	//weights might not always give a correct result, since I'm sharing finalSampleColour with all samples

	//Step 4: Sample 4 neighbours and take average
	RGBA rgba;
	RGBColor finalSampleColour{};
	for (int i{}; i < numNeighbours; ++i)
	{
		rgba = pixels[neighbours[i]];
		finalSampleColour.r += rgba.r8 * weight;
		finalSampleColour.g += rgba.g8 * weight;
		finalSampleColour.b += rgba.b8 * weight;
	}
	////sample the other 4 corners
	//finalSampleColour /= 2; //re-enable this bc of shared finalSampleColour
	//for (int i{4}; i < 8; ++i)
	//{
	//	SDL_GetRGB(rawData[(uint32_t)pixels[i]], m_pSurface->format, &r, &g, &b);
	//	finalSampleColour.r += r * weight2;
	//	finalSampleColour.g += g * weight2;
	//	finalSampleColour.b += b * weight2;
	//}
	//finalSampleColour /= 2;

	//Step 5: add original pixel sample and divide by 2 to not oversample colour
	rgba = pixels[originalPixel];
	finalSampleColour.r += rgba.r8 / 2.f;
	finalSampleColour.g += rgba.g8 / 2.f;
	finalSampleColour.b += rgba.b8 / 2.f;

	//Step 6: return finalSampleColour / 255
	return finalSampleColour / 255.f;
}