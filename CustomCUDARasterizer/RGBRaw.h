#pragma once
#include "GPUHelpers.h"
#include "RGBColor.h"

union RGBA
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

	struct
	{
		unsigned char r8, g8, b8, a8;
	};
	unsigned int colour32;

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

	BOTH_CALLABLE static
	RGBColor GetRGBColor(RGBA c)
	{
		return RGBColor{ c.r8 / 255.f, c.g8 / 255.f, c.b8 / 255.f };
	}

	BOTH_CALLABLE static
	RGBColor GetRGBColor_RBFlipped(RGBA c)
	{
		return RGBColor{ c.b8 / 255.f, c.g8 / 255.f, c.r8 / 255.f };
	}

	BOTH_CALLABLE static
	RGBColor GetRGBColor(unsigned int colour32)
	{
		return GetRGBColor(reinterpret_cast<RGBA&>(colour32));
	}

	BOTH_CALLABLE static
	RGBColor GetRGBColor_RBFlipped(unsigned int colour32)
	{
		return GetRGBColor_RBFlipped(reinterpret_cast<RGBA&>(colour32));
	}
};