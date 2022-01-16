#include "DirectionalLight.h"

DirectionalLight::DirectionalLight(const RGBColor& colour, float intensity, const FVector3& direction)
	: Light{colour, intensity}
	, m_Direction{direction}
{}

RGBColor DirectionalLight::GetBiradiance(const FPoint3& pointToShade) const
{
	return RGBColor{ m_Colour * m_Intensity };
}

FVector3 DirectionalLight::GetDirection(const FPoint3& pointToShade) const
{
	return m_Direction;
}

//float DirectionalLight::GetLambertCosineLaw(const HitRecord& hitRecord) const
//{
//	return Elite::Dot(hitRecord.normal, -m_Direction);
//}