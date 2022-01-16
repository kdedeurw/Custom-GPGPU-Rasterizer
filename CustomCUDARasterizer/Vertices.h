#pragma once
#include "Vertex.h"

struct Vertices
{
	//Vertices(const FPoint3& v0, const RGBColor& c0, const FPoint3& v1, const RGBColor& c1, const FPoint3& v2, const RGBColor& c2)
	//: v0{ v0, c0 }, v1{ v1, c1 }, v2{ v2, c2 } {}
	Vertices(const Vertex& v0, const Vertex& v1, const Vertex& v2)
	: v0{v0}, v1{v1}, v2{v2} {}

	Vertex v0;
	Vertex v1;
	Vertex v2;
};