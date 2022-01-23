#pragma once
#include "GPUHelpers.h"

#include "Math.h"
#include "MathUtilities.h"
#include "RGBColor.h"

struct IVertex_Base
{
	BOTH_CALLABLE IVertex_Base() //Purposely uninitialized
	{}
	BOTH_CALLABLE IVertex_Base(const FPoint4& position)
		: IVertex_Base{ FPoint3{ position } }
	{}
	BOTH_CALLABLE IVertex_Base(const FPoint3& position)
		: p{ position }
	{}
	BOTH_CALLABLE IVertex_Base(const IVertex_Base& other) // copy constructor => basically shallow copy
		: p{ other.p }
	{}

	FPoint3 p; // position
};

struct IVertex : public IVertex_Base
{
	BOTH_CALLABLE IVertex() //Purposely uninitialized
	{}
	BOTH_CALLABLE IVertex(const FPoint4& position,
		const FVector2& uvcoordinates, 
		const FVector3& normal = FVector3{1.f,1.f,1.f},
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex{ FPoint3{ position }, uvcoordinates, normal, colour }
	{}
	BOTH_CALLABLE IVertex(const FPoint3& position,
		const FVector2& uvcoordinates, 
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex_Base{ position }
		, uv{ uvcoordinates }
		, c{ colour }
		, n{ normal }
	{}
	BOTH_CALLABLE IVertex(const IVertex& other) // copy constructor => basically shallow copy
		: IVertex{ other.p, other.uv, other.n, other.c }
	{}

	RGBColor c; // colour
	FVector2 uv; // uv coordinate
	FVector3 n; // normal
	FVector3 tan; // tangent
};

struct OVertex_Base
{
	BOTH_CALLABLE OVertex_Base() //Purposely uninitialized
	{}
	BOTH_CALLABLE OVertex_Base(const FPoint3& position)
		: OVertex_Base{ FPoint4{position} }
	{}
	BOTH_CALLABLE OVertex_Base(const FPoint4& position)
		: p{ position }
	{}
	BOTH_CALLABLE OVertex_Base(const OVertex_Base& other) // copy constructor => basically shallow copy
		: p{ other.p }
	{}

	FPoint4 p; // position
};

struct OVertex : public OVertex_Base
{
	BOTH_CALLABLE OVertex() //Purposely uninitialized
	{}
	BOTH_CALLABLE OVertex(const FPoint3& position,
		const FVector2& uvcoordinates, 
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: OVertex{ FPoint4{position}, uvcoordinates, normal, FVector3{}, colour, FVector3{} }
	{}
	BOTH_CALLABLE OVertex(const FPoint3& position,
		const FVector2& uvcoordinates, 
		const FVector3& normal = FVector3{1.f,1.f,1.f},
		const FVector3& tangent = FVector3{},
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: OVertex{ FPoint4{position}, uvcoordinates, normal, tangent, colour, FVector3{} }
	{}
	BOTH_CALLABLE OVertex(const FPoint3& position,
		const FVector2& uvcoordinates, 
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const FVector3& tangent = FVector3{},
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f },
		const FVector3& viewDirection = FVector3{0.f,0.f,0.f})
		: OVertex{ FPoint4{position}, uvcoordinates, normal, tangent, colour, viewDirection }
	{}
	BOTH_CALLABLE OVertex(const FPoint4& position,
		const FVector2& uvcoordinates, 
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const FVector3& tangent = FVector3{},
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f },
		const FVector3& viewDirection = FVector3{ 0.f,0.f,0.f })
		: OVertex_Base{ position }
		, uv{ uvcoordinates }
		, c{ colour }
		, n{ normal }
		, tan{ tangent }
		, vd{viewDirection}
	{}
	BOTH_CALLABLE OVertex(const OVertex& other) // copy constructor => basically shallow copy
		: OVertex{ other.p, other.uv, other.n, other.tan, other.c, other.vd }
	{}

	RGBColor c; // colour
	FVector2 uv; // uv coordinate
	FVector3 n; // normal
	FVector3 tan; // tangent
	FVector3 vd;  // view direction
};