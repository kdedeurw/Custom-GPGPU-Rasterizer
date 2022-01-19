/*=============================================================================*/
// Copyright 2017-2019 Elite Engine
// Authors: Matthieu Delaere
/*=============================================================================*/
// ERGBColor.h: struct that represents a RGB color
/*=============================================================================*/
#ifndef ELITE_MATH_RGBCOLOR
#define	ELITE_MATH_RGBCOLOR

#include "GPUHelpers.h"

#include "MathUtilities.h"
#include <algorithm>

namespace Elite
{
	struct RGBColor final
	{
		//=== Datamembers ===
		float r = 0.f;
		float g = 0.f;
		float b = 0.f;

		//=== Constructors & Destructor ===
		BOTH_CALLABLE RGBColor() = default;
		BOTH_CALLABLE RGBColor(float _r, float _g, float _b) :r(_r), g(_g), b(_b) {}
		BOTH_CALLABLE RGBColor(const RGBColor& c) : r(c.r), g(c.g), b(c.b) {}
		BOTH_CALLABLE RGBColor(RGBColor&& c) noexcept : r(std::move(c.r)), g(std::move(c.g)), b(std::move(c.b)) {}
		BOTH_CALLABLE ~RGBColor() = default;

		//=== Operators ===
		BOTH_CALLABLE RGBColor& operator=(const RGBColor& c)
		{ r = c.r; g = c.g; b = c.b; return *this; }
		BOTH_CALLABLE RGBColor& operator=(RGBColor&& c) noexcept
		{ r = std::move(c.r); g = std::move(c.g); b = std::move(c.b); return *this;	}

		//=== Arithmetic Operators ===
		BOTH_CALLABLE inline RGBColor operator+(const RGBColor& c) const
		{ return RGBColor(r + c.r, g + c.g, b + c.b); }
		BOTH_CALLABLE inline RGBColor operator-(const RGBColor& c) const
		{ return RGBColor(r - c.r, g - c.g, b - c.b); }
		BOTH_CALLABLE inline RGBColor operator*(const RGBColor& c) const
		{ return RGBColor(r * c.r, g * c.g, b * c.b); }
		BOTH_CALLABLE inline RGBColor operator/(float f) const
		{
			float rev = 1.0f / f;
			return RGBColor(r * rev, g * rev, b * rev);
		}
		BOTH_CALLABLE inline RGBColor operator*(float f) const
		{ return RGBColor(r * f, g * f, b * f);	}
		BOTH_CALLABLE inline RGBColor operator/(const RGBColor& c) const
		{ return RGBColor(r / c.r, g / c.g, b / c.b); }

		//=== Compound Assignment Operators ===
		BOTH_CALLABLE inline RGBColor& operator+=(const RGBColor& c)
		{ r += c.r; g += c.g; b += c.b; return *this; }
		BOTH_CALLABLE inline RGBColor& operator-=(const RGBColor& c)
		{ r -= c.r; g -= c.g; b -= c.b; return *this; }
		BOTH_CALLABLE inline RGBColor& operator*=(const RGBColor& c)
		{ r *= c.r; g *= c.g; b *= c.b; return *this; }
		BOTH_CALLABLE inline RGBColor& operator/=(const RGBColor& c)
		{ r /= c.r; g /= c.g; b /= c.b; return *this; }
		BOTH_CALLABLE inline RGBColor& operator*=(float f)
		{ r *= f; g *= f; b *= f; return *this; }
		BOTH_CALLABLE inline RGBColor& operator/=(float f)
		{
			float rev = 1.0f / f;
			r *= rev; g *= rev; b *= rev; return *this;
		}

		//=== Internal RGBColor Functions ===
		BOTH_CALLABLE inline void ClampRGB()
		{
			r = Clamp(r, 0.0f, 1.0f);
			g = Clamp(g, 0.0f, 1.0f);
			b = Clamp(b, 0.0f, 1.0f);
		}

		BOTH_CALLABLE inline void MaxToOne()
		{
			float maxValue = std::max(r, std::max(g, b));
			if (maxValue > 1.f)
				*this /= maxValue;
		}
	};

	//=== Global RGBColor Functions ===
	BOTH_CALLABLE inline RGBColor Max(const RGBColor& c1, const RGBColor& c2)
	{
		RGBColor c = c1;
		if (c2.r > c.r) c.r = c2.r;
		if (c2.g > c.g) c.g = c2.g;
		if (c2.b > c.b) c.b = c2.b;
		return c;
	}

	BOTH_CALLABLE inline RGBColor Min(const RGBColor& c1, const RGBColor& c2)
	{
		RGBColor c = c1;
		if (c2.r < c.r) c.r = c2.r;
		if (c2.g < c.g) c.g = c2.g;
		if (c2.b < c.b) c.b = c2.b;
		return c;
	}
}
#endif