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
	const int x = uint32_t(uv.x * pTexture->w + 0.5f);
	const int y = uint32_t(uv.y * pTexture->h + 0.5f);
	if (x < 0 || y < 0 || x > pTexture->w || y > pTexture->h)
		return RGBColor{ 1.f, 0.f, 1.f };
	const uint32_t pixel = uint32_t(x + (y * pTexture->w));
	const uint32_t* pixels = (uint32_t*)pTexture->pixels;
	const RGBValues rgb = GetRGBFromTexture(pixels, pixel);
	return RGBColor{ rgb.r / 255.f, rgb.g / 255.f, rgb.b / 255.f };
}

GPU_CALLABLE GPU_INLINE RGBColor GPUTextureSampler::SampleLinear(const GPUTexture* pTexture, const FVector2& uv)
{
	//Step 1: find pixel to sample from
	const int x = int(uv.x * pTexture->w + 0.5f);
	const int y = int(uv.y * pTexture->h + 0.5f);
	if (x < 0 || y < 0 || x > pTexture->w || y > pTexture->h)
		return RGBColor{ 1.f, 0.f, 1.f };

	//Step 2: find # of neighbours
	const unsigned short numNeighbours = 4;
	const int originalPixel = int(x + (y * pTexture->w));
	const uint32_t texSize = pTexture->w * pTexture->h;

	int neighbourpixels[numNeighbours];
	//sample 4 adjacent neighbours
	neighbourpixels[0] = originalPixel - 1; //original pixel - 1x
	if (neighbourpixels[0] < 0)
		neighbourpixels[0] = 0;
	//possible issue: x might shove back from left to right - 1y
	neighbourpixels[1] = originalPixel + 1; //original pixel + 1x
	if (neighbourpixels[1] < texSize)
		neighbourpixels[1] = texSize;
	//possible issue: x might shove back from right to left + 1y
	neighbourpixels[2] = originalPixel + pTexture->w; //original pixel + 1y
	if (neighbourpixels[2] < texSize)
		neighbourpixels[2] = texSize;
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
	const float weight = 1.f / numNeighbours * 2; //# pixels, equally divided
	//weights might not always give a correct result, since I'm sharing finalSampleColour with all samples

	const uint32_t* pixels = (uint32_t*)pTexture->pixels;
	//Step 4: Sample 4 neighbours and take average
	RGBValues rgb;
	RGBColor finalSampleColour{};
	for (int i{}; i < numNeighbours; ++i)
	{
		rgb = GetRGBFromTexture(pixels, neighbourpixels[i]);
		finalSampleColour.r += rgb.r * weight;
		finalSampleColour.g += rgb.g * weight;
		finalSampleColour.b += rgb.b * weight;
	}
	////sample the other 4 corners
	//finalSampleColour /= 2; //re-enable this bc of shared finalSampleColour
	//for (int i{4}; i < 8; ++i)
	//{
	//	rgb = GetRGBFromTexture(pixels, neighbourpixels[i]);
	//	finalSampleColour.r += r * weight2;
	//	finalSampleColour.g += g * weight2;
	//	finalSampleColour.b += b * weight2;
	//}
	//finalSampleColour /= 2;

	//Step 5: add original pixel sample and divide by 2 to not oversample colour
	rgb = GetRGBFromTexture(pixels, originalPixel);
	finalSampleColour.r += rgb.r;
	finalSampleColour.g += rgb.g;
	finalSampleColour.b += rgb.b;
	finalSampleColour /= 2.f;

	//Step 6: return finalSampleColour / 255
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