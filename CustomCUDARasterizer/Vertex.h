#pragma once

#include "EMath.h"
#include "EMathUtilities.h"
#include "ERGBColor.h"
using namespace Elite;

struct IVertex
{
	IVertex(const FPoint4& position, const FVector2& uvcoordinates, const FVector3& normal = FVector3{1.f,1.f,1.f}, const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex{ FPoint3{ position }, uvcoordinates, normal, colour }
	{}
	IVertex(const FPoint3& position, const FVector2& uvcoordinates, const FVector3& normal = FVector3{ 1.f,1.f,1.f }, const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: v{ position }
		, uv{ uvcoordinates }
		, c{ colour }
		, n{ normal }
	{}

	FPoint3 v{}; // position
	RGBColor c{}; // colour
	FVector2 uv{}; // uv coordinate
	FVector3 n{}; // normal
	FVector3 tan{}; // tangent
};

struct OVertex
{
	OVertex(const FPoint3& position, const FVector2& uvcoordinates, const FVector3& normal = FVector3{ 1.f,1.f,1.f }, const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: OVertex{ FPoint4{position}, uvcoordinates, normal, FVector3{}, colour, FVector3{} }
	{}
	OVertex(const FPoint3& position, const FVector2& uvcoordinates, const FVector3& normal = FVector3{1.f,1.f,1.f}, const FVector3& tangent = FVector3{}, const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: OVertex{ FPoint4{position}, uvcoordinates, normal, tangent, colour, FVector3{} }
	{}
	OVertex(const FPoint3& position, const FVector2& uvcoordinates, const FVector3& normal = FVector3{ 1.f,1.f,1.f }, const FVector3& tangent = FVector3{}, const RGBColor& colour = RGBColor{ 1.f,1.f,1.f }, const FVector3& viewDirection = FVector3{0.f,0.f,0.f})
		: OVertex{ FPoint4{position}, uvcoordinates, normal, tangent, colour, viewDirection }
	{}
	OVertex(const FPoint4& position, const FVector2& uvcoordinates, const FVector3& normal = FVector3{ 1.f,1.f,1.f }, const FVector3& tangent = FVector3{}, const RGBColor& colour = RGBColor{ 1.f,1.f,1.f }, const FVector3& viewDirection = FVector3{ 0.f,0.f,0.f })
		: v{ position }
		, uv{ uvcoordinates }
		, c{ colour }
		, n{ normal }
		, tan{ tangent }
		, vd{viewDirection}
	{}
	
	OVertex(const OVertex& other) // copy constructor => basically shallow copy
		: v{ other.v }
		, c{ other.c }
		, uv{ other.uv }
		, n{ other.n }
		, tan{ other.tan }
		, vd{ other.vd }
	{}

	FPoint4 v{}; // position
	RGBColor c{}; // colour
	FVector2 uv{}; // uv coordinate
	FVector3 n{}; // normal
	FVector3 tan{}; // tangent
	FVector3 vd{};  // view direction
};