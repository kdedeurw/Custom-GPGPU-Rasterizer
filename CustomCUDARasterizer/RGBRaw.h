#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

#include "RGBColor.h"

//Forward declarations
struct RGB;
struct RGBA;

BOTH_CALLABLE RGBColor GetRGBColor(unsigned int colour);
BOTH_CALLABLE RGBColor GetRGBColor_RBFlipped(unsigned int colour);

struct RGBA
{
	//Uninitialized ctor
	BOTH_CALLABLE RGBA()
	{}
	BOTH_CALLABLE RGBA(unsigned int colour)
		: colour32{ colour }
	{}
	BOTH_CALLABLE RGBA(const RGBColor& colour)
		: colour32{ GetRGBAFromColour(colour).colour32 }
	{}

	union
	{
		struct
		{
			unsigned char r8, g8, b8, a8;
		};
		unsigned int colour32;
	};

	BOTH_CALLABLE static
	RGBA GetRGBAFromColour(const RGBColor& colour)
	{
		RGBA rgba;
		rgba.r8 = (unsigned char)(colour.b * 255);
		rgba.g8 = (unsigned char)(colour.g * 255);
		rgba.b8 = (unsigned char)(colour.r * 255);
		rgba.a8 = 0; //UCHAR_MAX // doesn't matter in this case
		return rgba;
	}
};

BOTH_CALLABLE RGBColor GetRGBColor(unsigned int colour)
{
	const RGBA& c = reinterpret_cast<RGBA&>(colour);
	return RGBColor{ c.r8 / 255.f, c.g8 / 255.f, c.b8 / 255.f };
}

BOTH_CALLABLE RGBColor GetRGBColor_RBFlipped(unsigned int colour)
{
	const RGBA& c = reinterpret_cast<RGBA&>(colour);
	return RGBColor{ c.b8 / 255.f, c.g8 / 255.f, c.r8 / 255.f };
}