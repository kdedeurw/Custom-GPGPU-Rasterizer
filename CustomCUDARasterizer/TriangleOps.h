#pragma once
#include "Math.h"
#include "MathUtilities.h"

namespace Triangle
{
	static FVector3 GetNormal(const FPoint3& p0, const FPoint3& p1, const FPoint3& p2);
	static float GetArea(const FPoint3& p0, const FPoint3& p1, const FPoint3& p2);
	static FPoint3 GetCenter(const FPoint3& p0, const FPoint3& p1, const FPoint3& p2);
};