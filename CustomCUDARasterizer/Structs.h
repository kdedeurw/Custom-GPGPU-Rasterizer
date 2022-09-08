#pragma once
#include "Math.h"
#include "RGBColor.h"
#include "BoundingBox.h"

struct RenderData // size == 35
{
	union
	{
		float pData[];
		struct
		{
			FPoint3 CamPos;
			FMatrix4 ViewProjectionMatrix;
			FMatrix4 WorldMatrix;
		};
	};
};

struct RasterTriangle // size == 12
{
	FPoint4 v0;
	FPoint4 v1;
	FPoint4 v2;
};

struct TriangleIdx // size == 3
{
	union
	{
		unsigned int idx[3];
		struct
		{
			unsigned int idx0;
			unsigned int idx1;
			unsigned int idx2;
		};
	};
};

struct TriangleIdxBb // size == 5
{
	union
	{
		unsigned int idx[];
		struct
		{
			unsigned int idx0;
			unsigned int idx1;
			unsigned int idx2;
		};
	};
	BoundingBox bb;
};