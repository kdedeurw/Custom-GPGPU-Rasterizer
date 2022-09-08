#include "PCH.h"
#include "TriangleOps.h"

FVector3 Triangle::GetNormal(const FPoint3& p0, const FPoint3& p1, const FPoint3& p2)
{
	const FVector3 a{ p1 - p0 };
	const FVector3 b{ p2 - p0 };
	return GetNormalized(Cross(a, b));
}

FPoint3 Triangle::GetCenter(const FPoint3& p0, const FPoint3& p1, const FPoint3& p2)
{
	const FVector3& v0 = reinterpret_cast<const FVector3&>(p0);
	const FVector3& v1 = reinterpret_cast<const FVector3&>(p1);
	const FVector3& v2 = reinterpret_cast<const FVector3&>(p2);
	return FPoint3{ v0 + v1 + v2 / 3 };
}

float Triangle::GetArea(const FPoint3& p0, const FPoint3& p1, const FPoint3& p2)
{
	std::initializer_list<float> yList{ p0.y, p1.y, p2.y };
	const float hMax = std::max(yList);
	const float hMin = std::min(yList);
	const float height = hMax - hMin;
	std::initializer_list<float> xList{ p0.x, p1.x, p2.x };
	const float bMax = std::max(xList);
	const float bMin = std::min(xList);
	const float base = bMax - bMin;

	return (height * base) / 2;
}