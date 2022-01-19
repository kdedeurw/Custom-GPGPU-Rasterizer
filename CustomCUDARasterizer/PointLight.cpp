#include "PCH.h"
#include "PointLight.h"

#include <iostream>

PointLight::PointLight(const RGBColor& colour, float intensity, const FPoint3& position)
	: Light{colour, intensity}
	, m_Position{position}
{}

RGBColor PointLight::GetBiradiance(const FPoint3& pointToShade) const
{
	return RGBColor{ m_Colour * (m_Intensity / SqrMagnitude(m_Position - pointToShade)) };
}

FVector3 PointLight::GetDirection(const FPoint3& pointToShade) const
{
	return GetNormalized(m_Position - pointToShade);
	//return GetNormalized(pointToShade - m_Position);
}

//float PointLight::GetLambertCosineLaw(const HitRecord& hitRecord) const
//{
//	return Dot(hitRecord.normal, GetDirection(hitRecord.point));
//}