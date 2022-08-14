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

struct IVertex_Colour : public IVertex_Base
{
	BOTH_CALLABLE IVertex_Colour() //Purposely uninitialized
	{}
	BOTH_CALLABLE IVertex_Colour(const FPoint4& position,
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex_Colour{ FPoint3{ position }, colour }
	{}
	BOTH_CALLABLE IVertex_Colour(const FPoint3& position,
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex_Base{ position }
		, c{ colour }
	{}
	BOTH_CALLABLE IVertex_Colour(const IVertex_Colour& other) // copy constructor => basically shallow copy
		: IVertex_Colour{ other.p, other.c }
	{}

	RGBColor c; // colour
};

struct IVertex_NoColour : public IVertex_Base
{
	BOTH_CALLABLE IVertex_NoColour() //Purposely uninitialized
	{}
	BOTH_CALLABLE IVertex_NoColour(const FPoint4& position,
		const FVector2& uvcoordinates,
		const FVector3& normal = FVector3{ 1.f,1.f,1.f })
		: IVertex_NoColour{ FPoint3{ position }, uvcoordinates, normal }
	{}
	BOTH_CALLABLE IVertex_NoColour(const FPoint3& position,
		const FVector2& uvcoordinates,
		const FVector3& normal = FVector3{ 1.f,1.f,1.f })
		: IVertex_Base{ position }
		, uv{ uvcoordinates }
		, n{ normal }
	{}
	BOTH_CALLABLE IVertex_NoColour(const IVertex_NoColour& other) // copy constructor => basically shallow copy
		: IVertex_NoColour{ other.p, other.uv, other.n }
	{}

	FVector2 uv; // uv coordinate
	FVector3 n; // normal
	FVector3 tan; // tangent
};

//struct IVertex
//{
//	BOTH_CALLABLE IVertex() //Purposely uninitialized
//	{}
//	BOTH_CALLABLE IVertex(const FPoint3& position,
//		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
//		: p{ position }
//		, c{colour}
//	{}
//	BOTH_CALLABLE IVertex(const FPoint4& position,
//		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
//		: IVertex{ FPoint3{ position }, colour }
//	{}
//	BOTH_CALLABLE IVertex(const FPoint3& position,
//		const FVector2& uvcoordinates,
//		const FVector3& normal = FVector3{ 1.f,1.f,1.f })
//		: p{ position }
//		, uv{ uvcoordinates }
//		, n{ normal }
//	{}
//	BOTH_CALLABLE IVertex(const FPoint4& position,
//		const FVector2& uvcoordinates,
//		const FVector3& normal = FVector3{ 1.f,1.f,1.f })
//		: IVertex{ FPoint3{ position }, uvcoordinates, normal }
//	{}
//	BOTH_CALLABLE ~IVertex()
//	{}
//	FPoint3 p;
//	union
//	{
//		struct
//		{
//			RGBColor c;
//		};
//		struct
//		{
//			FVector2 uv;
//			FVector3 n;
//			FVector3 tan;
//		};
//	};
//};

struct IVertex : public IVertex_Base
{
	BOTH_CALLABLE IVertex() //Purposely uninitialized
	{}
	BOTH_CALLABLE IVertex(const FPoint3& position,
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex_Base{ FPoint3{ position } }
		, c{ colour }
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

	FVector2 uv; // uv coordinate
	FVector3 n; // normal
	FVector3 tan; // tangent
	RGBColor c; // colour
};

struct IVertex_Point4
{
	BOTH_CALLABLE IVertex_Point4() //Purposely uninitialized
	{}
	BOTH_CALLABLE IVertex_Point4(const FPoint4& position,
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: p{ position }
		, c{ colour }
	{}
	BOTH_CALLABLE IVertex_Point4(const FPoint3& position,
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex_Point4{ FPoint4{ position }, colour }
	{}
	BOTH_CALLABLE IVertex_Point4(const FPoint4& position,
		const FVector2& uvcoordinates,
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: p{ position }
		, uv{ uvcoordinates }
		, c{ colour }
		, n{ normal }
	{}
	BOTH_CALLABLE IVertex_Point4(const FPoint3& position,
		const FVector2& uvcoordinates,
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const RGBColor& colour = RGBColor{ 1.f,1.f,1.f })
		: IVertex_Point4{ FPoint4{ position }, uvcoordinates, normal, colour }
	{}
	BOTH_CALLABLE IVertex_Point4(const IVertex& other) // copy constructor => basically shallow copy
		: IVertex_Point4{ other.p, other.uv, other.n, other.c }
	{}
	FPoint4 p; // position
	FVector2 uv; // uv coordinate
	FVector3 n; // normal
	FVector3 tan; // tangent
	RGBColor c; // colour
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

struct OVertex_Colour : public OVertex_Base
{
	BOTH_CALLABLE OVertex_Colour() //Purposely uninitialized
	{}
	BOTH_CALLABLE OVertex_Colour(const FPoint3& position, const RGBColor& colour)
		: OVertex_Base{ position }
		, c { colour }
	{}
	BOTH_CALLABLE OVertex_Colour(const FPoint4& position, const RGBColor& colour)
		: OVertex_Base{ position }
		, c{ colour }
	{}
	BOTH_CALLABLE OVertex_Colour(const OVertex_Colour& other) // copy constructor => basically shallow copy
		: OVertex_Colour{ other.p, other.c }
	{}

	RGBColor c; // colour
};

struct OVertex_NoColour : public OVertex_Base
{
	BOTH_CALLABLE OVertex_NoColour() //Purposely uninitialized
	{}
	BOTH_CALLABLE OVertex_NoColour(const FPoint3& position,
		const FVector2& uvcoordinates,
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const FVector3& tangent = FVector3{},
		const FVector3& viewDirection = FVector3{ 0.f,0.f,0.f })
		: OVertex_Base{ position }
		, uv { uvcoordinates }
		, n { normal }
		, tan { tangent }
		, vd { viewDirection }
	{}
	BOTH_CALLABLE OVertex_NoColour(const FPoint4& position,
		const FVector2& uvcoordinates,
		const FVector3& normal = FVector3{ 1.f,1.f,1.f },
		const FVector3& tangent = FVector3{},
		const FVector3& viewDirection = FVector3{ 0.f,0.f,0.f })
		: OVertex_Base{ position }
		, uv{ uvcoordinates }
		, n{ normal }
		, tan{ tangent }
		, vd{ viewDirection }
	{}
	BOTH_CALLABLE OVertex_NoColour(const OVertex_NoColour& other) // copy constructor => basically shallow copy
		: OVertex_NoColour{ other.p, other.uv, other.n, other.tan, other.vd }
	{}

	FVector2 uv; // uv coordinate
	FVector3 n; // normal
	FVector3 tan; // tangent
	FVector3 vd;  // view direction
};

//struct OVertex
//{
//	BOTH_CALLABLE OVertex() //Purposely uninitialized
//	{}
//	BOTH_CALLABLE ~OVertex()
//	{}
//	FPoint4 p;
//	union
//	{
//		struct
//		{
//			RGBColor c;
//		};
//		struct
//		{
//			FVector2 uv;
//			FVector3 n;
//			FVector3 tan;
//			FVector3 vd;
//		};
//	};
//};

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
		, vd{ viewDirection }
	{}
	BOTH_CALLABLE OVertex(const OVertex& other) // copy constructor => basically shallow copy
		: OVertex{ other.p, other.uv, other.n, other.tan, other.c, other.vd }
	{}

	FVector2 uv; // uv coordinate
	FVector3 n; // normal
	FVector3 tan; // tangent
	FVector3 vd;  // view direction
	RGBColor c; // colour
};

struct OVertexData // size == 14
{
	FVector2 uv;
	FVector3 n;
	FVector3 tan;
	FVector3 vd;
	RGBColor c;
};