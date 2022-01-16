#pragma once
#include "EMath.h"
#include "EMathUtilities.h"
#include "Vertex.h"
#include <array>

using namespace Elite;

class Triangle
{
public:
	//explicit Triangle(const FPoint3& position, const FPoint3& v0, const RGBColor& c0, const FPoint3& v1, const RGBColor& c1, const FPoint3& v2, const RGBColor& c2, bool isRotating = false, int cullMode = 0);
	explicit Triangle(const FPoint3& position, const IVertex& v0, const IVertex& v1, const IVertex& v2, bool isRotating = false, int cullMode = 0);
	explicit Triangle(const FPoint3& position, const std::array<IVertex, 3> vertices, bool isRotating = false, int cullMode = 0);
	virtual ~Triangle() = default;

	const std::array<IVertex, 3> &GetVertices() const;
	const FVector3& GetNormal() const;

	float GetArea() const;

	float CalculateArea() const;

private:
	bool m_IsRotating;
	float m_Area;
	FPoint3 m_Position;
	std::array<IVertex, 3> m_Vertices;
	FVector3 m_Normal;
	FPoint3 m_Center;

	enum class Cullmode
	{
		backface,
		frontface,
		noculling
	};
	Cullmode m_Cullmode;
};