#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

#include "Math.h"
#include "RGBColor.h"
#include "SampleState.h"
#include "MathUtilities.h"
#include "GPUTextures.h"
#include "RGBRaw.h"

namespace GPUTextureSampler
{
	GPU_CALLABLE GPU_INLINE RGBColor Sample(const GPUTexture* pTexture, const FVector2& uv, const SampleState sampleState);
	GPU_CALLABLE GPU_INLINE RGBColor SamplePoint(const GPUTexture* pTexture, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBColor SampleLinear(const GPUTexture* pTexture, const FVector2& uv);
	GPU_CALLABLE GPU_INLINE RGBValues GetRGBFromTexture(const uint32_t* pPixels, uint32_t pixelIdx);
};

GPU_CALLABLE GPU_INLINE RGBColor GPUTextureSampler::Sample(const GPUTexture* pTexture, const FVector2& uv, const SampleState sampleState)
{
	if (sampleState == SampleState::Point)
		return SamplePoint(pTexture, uv);
	return SampleLinear(pTexture, uv);
}

GPU_CALLABLE GPU_INLINE RGBColor GPUTextureSampler::SamplePoint(const GPUTexture* pTexture, const FVector2& uv)
{
	const uint32_t x{ uint32_t(uv.x * pTexture->w) };
	const uint32_t y{ uint32_t(uv.y * pTexture->h) };
	const uint32_t pixel = uint32_t(x + (y * pTexture->w));
	//const SDL_PixelFormat* pPixelFormat = new SDL_PixelFormat{ m_pSurface->format->BytesPerPixel };
	const uint32_t* pixels = (uint32_t*)pTexture->pixels; // only works with uint32*, as seen in Renderer.h
	//SDL_GetRGB(pixels[pixel], m_pSurface->format, &r, &g, &b);
	const RGBValues rgb = GetRGBFromTexture(pixels, pixel);
	return RGBColor{ rgb.r / 255.f, rgb.g / 255.f, rgb.b / 255.f };
}

GPU_CALLABLE GPU_INLINE RGBColor GPUTextureSampler::SampleLinear(const GPUTexture* pTexture, const FVector2& uv)
{
	//Step 1: find pixel to sample from
	const uint32_t x{ uint32_t(uv.x * pTexture->w) };
	const uint32_t y{ uint32_t(uv.y * pTexture->h) };

	//Step 2: find 4 neighbours
	const uint32_t* rawData = (uint32_t*)pTexture->pixels;
	const uint32_t max = pTexture->w * pTexture->h;
	const uint32_t originalPixel = uint32_t(x + (y * pTexture->w));

	int neighbourpixels[4];
	//sample 4 adjacent neighbours
	neighbourpixels[0] = originalPixel - 1; //original pixel - 1x
	if (neighbourpixels[0] < 0)
		neighbourpixels[0] = 0;
	//possible issue: x might shove back from left to right - 1y
	neighbourpixels[1] = originalPixel + 1; //original pixel + 1x
	if (neighbourpixels[1] < max)
		neighbourpixels[1] = max;
	//possible issue: x might shove back from right to left + 1y
	neighbourpixels[2] = originalPixel + pTexture->w; //original pixel + 1y
	if (neighbourpixels[2] < max)
		neighbourpixels[2] = max;
	neighbourpixels[3] = originalPixel - pTexture->w; //original pixel - 1y
	if (neighbourpixels[3] < 0)
		neighbourpixels[3] = 0;

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
	const float weight = 0.5f; //4 pixels, equally divided
	//const float weight2 = 0.25f; //4 pixels, equally divided, but count for half as adjacent ones
	//weights might not always give a correct result, since I'm sharing finalSampleColour with all samples

	//Step 4: Sample 4 neighbours and take average
	RGBValues rgb;
	RGBColor finalSampleColour{};
	for (int i{}; i < 4; ++i)
	{
		//SDL_GetRGB(rawData[neighbourpixels[i]], m_pSurface->format, &r, &g, &b);
		rgb = GetRGBFromTexture(rawData, neighbourpixels[i]);
		finalSampleColour.r += rgb.r * weight;
		finalSampleColour.g += rgb.g * weight;
		finalSampleColour.b += rgb.b * weight;
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

	//SDL_GetRGB(rawData[originalPixel], m_pSurface->format, &r, &g, &b);
	rgb = GetRGBFromTexture(rawData, originalPixel);
	finalSampleColour.r += rgb.r;
	finalSampleColour.g += rgb.g;
	finalSampleColour.b += rgb.b;
	finalSampleColour /= 2;

	//Step 5: return finalSampleColour / 255
	return finalSampleColour / 255.f;
}

//Unecessary copy
GPU_CALLABLE GPU_INLINE RGBValues GPUTextureSampler::GetRGBFromTexture(const uint32_t* pPixels, uint32_t pixelIdx)
{
	const unsigned char* pRGB = (unsigned char*)pPixels[pixelIdx];
	RGBValues rgb;
	rgb.r = pRGB[0];
	rgb.g = pRGB[1];
	rgb.b = pRGB[2];
	return rgb;
}