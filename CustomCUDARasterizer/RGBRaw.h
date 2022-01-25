#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

#include "RGBColor.h"

//Forward declarations
struct RGBValues;
struct RGBAValues;
union RGB;
union RGBA;
BOTH_CALLABLE RGB GetRGB_SDL(const RGBColor& colour);
BOTH_CALLABLE RGBA GetRGBA_SDL(const RGBColor& colour);

struct RGBValues
{
	//Uninitialized ctor
	BOTH_CALLABLE RGBValues()
	{}
	BOTH_CALLABLE RGBValues(unsigned char r, unsigned char g, unsigned char b)
		: r{ r }
		, g{ g }
		, b{ b }
	{}
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

struct RGBAValues : public RGBValues
{
	//Uninitialized ctor
	BOTH_CALLABLE RGBAValues()
	{}
	BOTH_CALLABLE RGBAValues(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
		: RGBValues{ r, g, b }
		, a{ a }
	{}
	unsigned char a;
};

union RGB
{
	//Uninitialized ctor
	BOTH_CALLABLE RGB()
	{}
	BOTH_CALLABLE RGB(unsigned int colour)
		: colour{ colour }
	{}
	BOTH_CALLABLE RGB(const RGBColor& colour)
		: colour{ GetRGB_SDL(colour).colour }
	{}
	RGBValues values;
	unsigned int colour;
};

union RGBA
{
	//Uninitialized ctor
	BOTH_CALLABLE RGBA()
	{}
	BOTH_CALLABLE RGBA(unsigned int colour)
		: colour{ colour }
	{}
	BOTH_CALLABLE RGBA(const RGBColor& colour)
		: colour{ GetRGBA_SDL(colour).colour }
	{}
	RGBAValues values;
	unsigned int colour;
};

BOTH_CALLABLE RGB GetRGB_SDL(const RGBColor& colour)
{
	RGB rgb;
	rgb.values.r = (unsigned char)(colour.b * 255);
	rgb.values.g = (unsigned char)(colour.g * 255);
	rgb.values.b = (unsigned char)(colour.r * 255);
	return rgb;
}

BOTH_CALLABLE RGBA GetRGBA_SDL(const RGBColor& colour)
{
	RGBA rgba;
	rgba.values.r = (unsigned char)(colour.b * 255);
	rgba.values.g = (unsigned char)(colour.g * 255);
	rgba.values.b = (unsigned char)(colour.r * 255);
	rgba.values.a = 0; //UCHAR_MAX // doesn't matter in this case
	return rgba;
}

//TODO: static_cast + return, without copying into RGBA struct
BOTH_CALLABLE RGBColor GetRGBColor_SDL(unsigned int colour)
{
	const RGBA c{ std::move(colour) };
	return RGBColor{ c.values.r / 255.f, c.values.g / 255.f, c.values.b / 255.f };
}