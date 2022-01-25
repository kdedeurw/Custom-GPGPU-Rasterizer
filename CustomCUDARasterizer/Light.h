#pragma once
#include "RGBColor.h"
#include "Math.h"
#include "GPUHelpers.h"

class Light
{
public:
	explicit Light(const RGBColor& colour, float intensity);
	virtual ~Light() = default;

	BOTH_CALLABLE virtual RGBColor GetBiradiance(const FPoint3& pointToShade) const = 0;
	BOTH_CALLABLE virtual FVector3 GetDirection(const FPoint3& pointToShade) const = 0;
	BOTH_CALLABLE virtual float GetIntensity() const { return m_Intensity; }

	//virtual float GetLambertCosineLaw(const HitRecord& hitRecord) const = 0;

protected:
	float m_Intensity;
	RGBColor m_Colour;
};

