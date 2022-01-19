#pragma once
#include "RGBColor.h"
#include "Math.h"

class Light
{
public:
	explicit Light(const RGBColor& colour, float intensity);
	virtual ~Light() = default;

	virtual RGBColor GetBiradiance(const FPoint3& pointToShade) const = 0;
	virtual FVector3 GetDirection(const FPoint3& pointToShade) const = 0;

	//virtual float GetLambertCosineLaw(const HitRecord& hitRecord) const = 0;

protected:
	float m_Intensity;
	RGBColor m_Colour;
};

