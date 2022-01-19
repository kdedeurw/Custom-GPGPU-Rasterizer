#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

#include "Math.h"
#include "RGBColor.h"
#include "SampleState.h"
#include "MathUtilities.h"
#include <SDL_image.h>
#include <SDL_surface.h>

struct RGB
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

struct SDL_Surface;
class GPUTexture
{
public:
	CPU_CALLABLE explicit GPUTexture(const char* file);
	CPU_CALLABLE ~GPUTexture();
	GPU_CALLABLE inline RGBColor Sample(const FVector2& uv, SampleState state = SampleState::Point) const;

private:
	SDL_Surface* m_pSurface;

	GPU_CALLABLE inline RGBColor SamplePoint(const FVector2& uv) const;
	GPU_CALLABLE inline RGBColor SampleLinear(const FVector2& uv) const;
	GPU_CALLABLE inline RGB GetRGBFromTexture(const uint32_t* pPixels, uint32_t pixelIdx) const;
};

GPU_CALLABLE __forceinline__ RGBColor GPUTexture::Sample(const FVector2& uv, SampleState state) const
{
	if (state == SampleState::Linear)
		return SampleLinear(uv);

	return SamplePoint(uv);
}

GPU_CALLABLE RGBColor GPUTexture::SamplePoint(const FVector2& uv) const
{
	const uint32_t x{ uint32_t(uv.x * m_pSurface->w) };
	const uint32_t y{ uint32_t(uv.y * m_pSurface->h) };
	const uint32_t pixel = uint32_t(x + (y * m_pSurface->w));
	//const SDL_PixelFormat* pPixelFormat = new SDL_PixelFormat{ m_pSurface->format->BytesPerPixel };
	const uint32_t* pixels = (uint32_t*)m_pSurface->pixels; // only works with uint32*, as seen in Renderer.h
	//SDL_GetRGB(pixels[pixel], m_pSurface->format, &r, &g, &b);
	const RGB rgb = GetRGBFromTexture(pixels, pixel);
	return RGBColor{ rgb.r / 255.f, rgb.g / 255.f, rgb.b / 255.f };
}

GPU_CALLABLE RGBColor GPUTexture::SampleLinear(const FVector2& uv) const
{
	//Step 1: find pixel to sample from
	const uint32_t x{ uint32_t(uv.x * m_pSurface->w) };
	const uint32_t y{ uint32_t(uv.y * m_pSurface->h) };

	//Step 2: find 4 neighbours
	const uint32_t* rawData = (uint32_t*)m_pSurface->pixels;
	const uint32_t max = m_pSurface->w * m_pSurface->h;
	const uint32_t originalPixel = uint32_t(x + (y * m_pSurface->w));

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
	neighbourpixels[2] = originalPixel + m_pSurface->w; //original pixel + 1y
	if (neighbourpixels[2] < max)
		neighbourpixels[2] = max;
	neighbourpixels[3] = originalPixel - m_pSurface->w; //original pixel - 1y
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
	RGB rgb;
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

GPU_CALLABLE inline RGB GPUTexture::GetRGBFromTexture(const uint32_t* pPixels, uint32_t pixelIdx) const
{
	const Uint8* pRGB = (Uint8*)pPixels[pixelIdx];
	RGB rgb;
	rgb.r = pRGB[0];
	rgb.g = pRGB[1];
	rgb.b = pRGB[2];
	return rgb;
}