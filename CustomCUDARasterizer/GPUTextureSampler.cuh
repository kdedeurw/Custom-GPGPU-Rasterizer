#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

#include "RGBColor.h"
#include "Math.h"
#include "GPUTextures.h"
#include "RGBRaw.h"
#include "SampleState.h"

namespace GPUTextureSampler
{
	GPU_CALLABLE GPU_INLINE RGBColor Sample1D(const GPUTexture& gpuTexture, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor Sample2D(const GPUTexture& gpuTexture, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor Sample(const GPUTexture& gpuTexture, const FVector2& uv, SampleState sampleState);
	GPU_CALLABLE GPU_INLINE RGBColor SamplePoint(const GPUTexture& gpuTexture, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor SampleLinear(const GPUTexture& gpuTexture, const FVector2& uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor GPUTextureSampler::Sample(const GPUTexture& gpuTexture, const FVector2& uv, SampleState sampleState)
{
	if (sampleState == SampleState::Linear)
		return GPUTextureSampler::SampleLinear(gpuTexture, uv);

	return GPUTextureSampler::SamplePoint(gpuTexture, uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor GPUTextureSampler::Sample1D(const GPUTexture& gpuTexture, const FVector2& uv)
{
	//// Transform coordinates
	float u{ uv.x };
	float v{ uv.y };
	u *= gpuTexture.w;
	v *= gpuTexture.h;
	//u += 0.5f;
	//v += 0.5f;
	//u -= 0.5f;
	//v -= 0.5f;
	//float tu = u * cosf(theta) ? v * sinf(theta) + 0.5f;
	//float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

	unsigned int sample = tex1Dfetch<unsigned int>(gpuTexture.dev_pTex, u + v * gpuTexture.w);
	const RGBA rgba{ sample };
	return RGBColor{ rgba.values.b / 255.f, rgba.values.g / 255.f, rgba.values.r / 255.f };
}

GPU_CALLABLE GPU_INLINE
RGBColor GPUTextureSampler::Sample2D(const GPUTexture& gpuTexture, const FVector2& uv)
{
	//// Transform coordinates
	float u{ uv.x };
	float v{ uv.y };
	//u += 0.5f;
	//v += 0.5f;
	//u -= 0.5f;
	//v -= 0.5f;
	//float tu = u * cosf(theta) ? v * sinf(theta) + 0.5f;
	//float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

	unsigned int sample = tex2D<unsigned int>(gpuTexture.dev_pTex, u, v);
	const RGBA rgba{ sample };
	return RGBColor{ rgba.values.b / 255.f, rgba.values.g / 255.f, rgba.values.r / 255.f };
}

GPU_CALLABLE GPU_INLINE
RGBColor GPUTextureSampler::SamplePoint(const GPUTexture& gpuTexture, const FVector2& uv)
{
	// Transform coordinates
	const int u{ int(uv.x * gpuTexture.w + 0.5f) };
	const int v{ int(uv.y * gpuTexture.h + 0.5f) };
	if (u < 0 || v < 0 || u >= gpuTexture.w || v >= gpuTexture.h)
		return RGBColor{ 1.f, 0.f, 1.f };
	const int sampleIdx = (int)u + (int)v * gpuTexture.w;
	unsigned int sample = gpuTexture.dev_TextureData[sampleIdx];
	RGBColor sampleColour = GetRGBColor_SDL(sample);
	return sampleColour;
}

GPU_CALLABLE GPU_INLINE
RGBColor GPUTextureSampler::SampleLinear(const GPUTexture& gpuTexture, const FVector2& uv)
{
	//Step 1: find pixel to sample from
	const int x = int(uv.x * gpuTexture.w + 0.5f);
	const int y = int(uv.y * gpuTexture.h + 0.5f);
	if (x < 0 || y < 0 || x > gpuTexture.w || y > gpuTexture.h)
		return RGBColor{ 1.f, 0.f, 1.f };

	//Step 2: find 4 neighbours
	const unsigned int* pixels = gpuTexture.dev_TextureData;
	const int texSize = gpuTexture.w * gpuTexture.h;
	const int originalPixel = x + y * gpuTexture.w;

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
	neighbours[2] = originalPixel + gpuTexture.w; //original pixel + 1y
	if (neighbours[2] >= texSize)
		neighbours[2] = originalPixel;
	neighbours[3] = originalPixel - gpuTexture.w; //original pixel - 1y
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
		finalSampleColour.r += rgba.values.r * weight;
		finalSampleColour.g += rgba.values.g * weight;
		finalSampleColour.b += rgba.values.b * weight;
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
	finalSampleColour.r += rgba.values.r / 2.f;
	finalSampleColour.g += rgba.values.g / 2.f;
	finalSampleColour.b += rgba.values.b / 2.f;

	//Step 6: return finalSampleColour / 255
	return finalSampleColour / 255.f;
}