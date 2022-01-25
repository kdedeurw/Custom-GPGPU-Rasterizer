#pragma once
#include "Light.h"

class DirectionalLight : public Light
{
public:
	explicit DirectionalLight(const RGBColor& colour, float intensity, const FVector3& direction);
	virtual ~DirectionalLight() = default;

	BOTH_CALLABLE virtual RGBColor GetBiradiance(const FPoint3& pointToShade) const override;
	BOTH_CALLABLE virtual FVector3 GetDirection(const FPoint3& pointToShade) const override;

	//virtual float GetLambertCosineLaw(const HitRecord& hitRecord) const override;

private:
	FVector3 m_Direction;
};

