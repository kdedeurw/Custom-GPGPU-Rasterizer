#pragma once
#include "Light.h"

class PointLight : public Light
{
public:
	explicit PointLight(const RGBColor& colour, float intensity, const FPoint3& position);
	virtual ~PointLight() = default;

	virtual RGBColor GetBiradiance(const FPoint3& pointToShade) const override;
	virtual FVector3 GetDirection(const FPoint3& pointToShade) const override;
	//virtual float GetLambertCosineLaw(const HitRecord& hitRecord) const override;

private:
	FPoint3 m_Position;
};

