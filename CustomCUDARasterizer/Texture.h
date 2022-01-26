#pragma once
#include "Math.h"
#include "RGBColor.h"
#include "SampleState.h"
#include "MathUtilities.h"
#include <SDL_image.h>
#include <SDL_surface.h>
#include "Textures.h"

struct SDL_Surface;
class Texture
{
public:
	explicit Texture(const char* file);
	~Texture();
	RGBColor inline Sample(const FVector2& uv, SampleState state = SampleState::Point) const;
	RGBColor inline SamplePoint(const FVector2& uv) const;
	RGBColor inline SampleLinear(const FVector2& uv) const;

private:
	SDL_Surface* m_pSurface;

};

RGBColor Texture::Sample(const FVector2& uv, SampleState state) const
{
	if (state == SampleState::Linear)
		return SampleLinear(uv);

	return SamplePoint(uv);
}

RGBColor Texture::SamplePoint(const FVector2& uv) const
{
	const int x = int(uv.x * m_pSurface->w + 0.5f);
	const int y = int(uv.y * m_pSurface->h + 0.5f);
	if (x < 0 || y < 0 || x > m_pSurface->w || y > m_pSurface->h)
		return RGBColor{ 1.f, 0.f, 1.f };
	Uint8 r, g, b;
	const uint32_t pixel = x + y * m_pSurface->w;
	uint32_t* pixels = (uint32_t*)m_pSurface->pixels;
	SDL_GetRGB(pixels[pixel], m_pSurface->format, &r, &g, &b);
	return RGBColor{ r / 255.f, g / 255.f, b / 255.f };
}

RGBColor Texture::SampleLinear(const FVector2& uv) const
{
	//Step 1: find pixel to sample from
	const int x = int(uv.x * m_pSurface->w);
	const int y = int(uv.y * m_pSurface->h);
	if (x < 0 || y < 0 || x > m_pSurface->w || y > m_pSurface->h)
		return RGBColor{ 1.f, 0.f, 1.f };

	//Step 2: find 4 neighbours
	const uint32_t* pixels = (uint32_t*)m_pSurface->pixels;
	const int texSize = m_pSurface->w * m_pSurface->h;
	const int originalPixel = uint32_t(x + (y * m_pSurface->w));

	const unsigned short numNeighbours = 4;
	int neighbours[numNeighbours];
	//sample 4 adjacent neighbours
	neighbours[0] = x - 1 + (y * m_pSurface->w); //original pixel - 1x
	if (neighbours[0] < 0)
		neighbours[0] = 0;
	//possible issue: x might shove back from left to right - 1y
	neighbours[1] = x + 1 + (y * m_pSurface->w); //original pixel + 1x
	if (neighbours[1] < texSize)
		neighbours[1] = texSize;
	//possible issue: x might shove back from right to left + 1y
	neighbours[2] = x + ((y + 1) * m_pSurface->w); //original pixel + 1y
	if (neighbours[2] < texSize)
		neighbours[2] = texSize;
	neighbours[3] = x + ((y - 1) * m_pSurface->w); //original pixel - 1y
	if (neighbours[3] < 0)
		neighbours[3] = 0;

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

	//Step 4: Sample 4 neighbours and take average
	Uint8 r, g, b;
	RGBColor finalSampleColour{};
	for (int i{}; i < numNeighbours; ++i)
	{
		SDL_GetRGB(pixels[(uint32_t)neighbours[i]], m_pSurface->format, &r, &g, &b);
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

	//Step 5: add original pixel sample and divide by 2 to not oversample colour
	SDL_GetRGB(pixels[originalPixel], m_pSurface->format, &r, &g, &b);
	finalSampleColour.r += r;
	finalSampleColour.g += g;
	finalSampleColour.b += b;
	finalSampleColour /= 2.f;

	//Step 6: return finalSampleColour / 255
	return finalSampleColour / 255.f;
}