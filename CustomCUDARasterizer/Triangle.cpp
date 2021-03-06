#include "PCH.h"
#include "Triangle.h"

//Triangle::Triangle(const FPoint3& position, const FPoint3& v0, const RGBColor& c0, const FPoint3& v1, const RGBColor& c1, const FPoint3& v2, const RGBColor& c2, bool isRotating, int cullMode)
//	: Triangle{ position, Vertex{v0, c0}, Vertex{v1, c1}, Vertex{v2, c2}, isRotating, cullMode }
//{
//}

Triangle::Triangle(const FPoint3& position, const std::array<IVertex, 3> vertices, bool isRotating, int cullMode)
	: Triangle{ position, vertices[0], vertices[1], vertices[2], isRotating, cullMode }
{
}

Triangle::Triangle(const FPoint3& position, const IVertex& v0, const IVertex& v1, const IVertex& v2, bool isRotating, int cullMode)
	: m_Position{ position }
	, m_Cullmode{ Cullmode::noculling }
	, m_Vertices{v0, v1, v2}
	, m_IsRotating{isRotating}
	, m_Normal{ GetNormalized(Cross(FVector3{v1.p - v0.p}, FVector3{v2.p - v0.p})) }
	, m_Center{ FPoint3{ (FVector3{v0.p.x, v0.p.y, v0.p.z} +FVector3{v1.p.x, v1.p.y, v1.p.z} +FVector3{v2.p.x, v2.p.y, v2.p.z}) / 3 } }
	, m_Area{ CalculateArea() }
{
	FVector3 a{ v1.p - v0.p };
	FVector3 b{ v2.p - v0.p };
	m_Normal = GetNormalized(Cross(a, b));
	m_Center = FPoint3{ (FVector3{v0.p.x, v0.p.y, v0.p.z} +FVector3{v1.p.x, v1.p.y, v1.p.z} +FVector3{v2.p.x, v2.p.y, v2.p.z}) / 3 };

	switch (cullMode)
	{
	case 0:
		m_Cullmode = Cullmode::noculling;
		break;
	case 1:
		m_Cullmode = Cullmode::backface;
		break;
	case 2:
		m_Cullmode = Cullmode::frontface;
		break;
	default:
		m_Cullmode = Cullmode::noculling;
		break;
	}
}

const std::array<IVertex, 3>& Triangle::GetVertices() const
{
	return m_Vertices;
}

const FVector3& Triangle::GetNormal() const
{
	return m_Normal;
}

float Triangle::GetArea() const
{
	return m_Area;
}

float Triangle::CalculateArea() const
{
	//std::initializer_list<float> yList{ m_Vertices.v0.v.y, m_Vertices.v1.v.y, m_Vertices.v2.v.y };
	std::initializer_list<float> yList{ m_Vertices[0].p.y, m_Vertices[1].p.y, m_Vertices[2].p.y };
	float hMax{ std::max(yList) };
	float hMin{ std::min(yList) };
	float height{ hMax - hMin };
	//std::initializer_list<float> xList{ m_Vertices.v0.v.x, m_Vertices.v1.v.x, m_Vertices.v2.v.x };
	std::initializer_list<float> xList{ m_Vertices[0].p.x, m_Vertices[1].p.x, m_Vertices[2].p.x };
	float bMax{ std::max(xList) };
	float bMin{ std::min(xList) };
	float base{ bMax - bMin };

	return (height * base) / 2;
}