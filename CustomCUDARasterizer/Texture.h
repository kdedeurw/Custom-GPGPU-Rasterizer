#pragma once
#include "Math.h"
#include "RGBColor.h"
#include "SampleState.h"
#include "MathUtilities.h"
#include <SDL_image.h>
#include <SDL_surface.h>

struct SDL_Surface;
class Texture
{
public:
	explicit Texture(const char* file);
	~Texture();
	RGBColor inline Sample(const FVector2& uv, SampleState state = SampleState::Point) const;

private:
	SDL_Surface* m_pSurface;

	RGBColor inline SamplePoint(const FVector2& uv) const;
	RGBColor inline SampleLinear(const FVector2& uv) const;
};

RGBColor Texture::Sample(const FVector2& uv, SampleState state) const
{
	if (state == SampleState::Linear)
		return SampleLinear(uv);

	return SamplePoint(uv);
}

RGBColor Texture::SamplePoint(const FVector2& uv) const
{
	Uint8 r{}, g{}, b{};
	//SDL_Color colour{};
	const uint32_t x{ uint32_t(uv.x * m_pSurface->w) };
	const uint32_t y{ uint32_t(uv.y * m_pSurface->h) };
	const uint32_t pixel = uint32_t(x + (y * m_pSurface->w));
	//const SDL_PixelFormat* pPixelFormat = new SDL_PixelFormat{ m_pSurface->format->BytesPerPixel };
	const uint32_t* pixels = (uint32_t*)m_pSurface->pixels; // only works with uint32*, as seen in Renderer.h
	SDL_GetRGB(pixels[pixel], m_pSurface->format, &r, &g, &b);
	//SDL_GetRGBA(pixels[pixel], m_pSurface->format, &colour.r, &colour.g, &colour.b, &colour.a);
	return RGBColor{ r / 255.f, g / 255.f, b / 255.f };
	//return RGBColor{ float(colour.r), float(colour.g), float(colour.b) } / 255.f;
}

RGBColor Texture::SampleLinear(const FVector2& uv) const
{
	//Step 1: find pixel to sample from
	const uint32_t x{ uint32_t(uv.x * m_pSurface->w) };
	const uint32_t y{ uint32_t(uv.y * m_pSurface->h) };

	//Step 2: find 4 neighbours
	const uint32_t* rawData = (uint32_t*)m_pSurface->pixels;
	const uint32_t max = m_pSurface->w * m_pSurface->h;
	const uint32_t originalPixel = uint32_t(x + (y * m_pSurface->w));

	int pixels[4];
	//sample 4 adjacent neighbours
	pixels[0] = x - 1 + (y * m_pSurface->w); //original pixel - 1x
	if (pixels[0] < 0)
		pixels[0] = 0;
	//possible issue: x might shove back from left to right - 1y
	pixels[1] = x + 1 + (y * m_pSurface->w); //original pixel + 1x
	if (pixels[1] < max)
		pixels[1] = max;
	//possible issue: x might shove back from right to left + 1y
	pixels[2] = x + ((y + 1) * m_pSurface->w); //original pixel + 1y
	if (pixels[2] < max)
		pixels[2] = max;
	pixels[3] = x + ((y - 1) * m_pSurface->w); //original pixel - 1y
	if (pixels[3] < 0)
		pixels[3] = 0;

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
	Uint8 r{}, g{}, b{};
	RGBColor finalSampleColour{};
	for (int i{}; i < 4; ++i)
	{
		SDL_GetRGB(rawData[(uint32_t)pixels[i]], m_pSurface->format, &r, &g, &b);
		finalSampleColour.r += r * weight;
		finalSampleColour.g += g * weight;
		finalSampleColour.b += b * weight;
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

	SDL_GetRGB(rawData[originalPixel], m_pSurface->format, &r, &g, &b);
	finalSampleColour.r += r;
	finalSampleColour.g += g;
	finalSampleColour.b += b;
	finalSampleColour /= 2;

	//Step 5: return finalSampleColour / 255
	return finalSampleColour / 255.f;
}