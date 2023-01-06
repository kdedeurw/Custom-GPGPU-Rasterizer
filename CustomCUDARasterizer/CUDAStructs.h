#pragma once
#include "Math.h"
#include "RGBColor.h"
#include "CUDATexture.h"
#include "Vertex.h"

struct CUDATextureCompact
{
	static CUDATextureCompact CompactCUDATexture(const CUDATexture& tex)
	{
		return CUDATextureCompact{ tex };
	}

	CUDATextureCompact() = default;
	CUDATextureCompact(const CUDATextureCompact& other) = default;
	CUDATextureCompact(CUDATextureCompact&& other) = default;
	CUDATextureCompact& operator=(const CUDATextureCompact& other) = default;
	CUDATextureCompact& operator=(CUDATextureCompact&& other) = default;
	CUDATextureCompact& operator=(const CUDATexture& tex)
	{
		this->w = tex.GetWidth();
		this->h = tex.GetHeight();
		this->dev_pTex = tex.GetTextureObject();
		this->dev_TextureData = tex.GetTextureData();
		return *this;
	};
	explicit CUDATextureCompact(const CUDATexture& tex)
		: w{ tex.GetWidth() }
		, h{ tex.GetHeight() }
		, dev_pTex{ tex.GetTextureObject() }
		, dev_TextureData{ tex.GetTextureData() }
	{};
	unsigned int w; //width
	unsigned int h; //height
	cudaTextureObject_t dev_pTex; //cuda texture pointer address
	unsigned int* dev_TextureData; //pixel array
};

struct CUDATexturesCompact
{
	CUDATextureCompact Diff;
	CUDATextureCompact Norm;
	CUDATextureCompact Spec;
	CUDATextureCompact Gloss;
};

struct PixelShade
{
	unsigned int colour32;
	float zInterpolated;
	float wInterpolated;
	FVector2 uv;
	FVector3 n;
	FVector3 tan;
	FVector3 vd;
	CUDATexturesCompact textures;
};

struct OVertex_PosShared
{
	BOTH_CALLABLE
	OVertex_PosShared() {};
	BOTH_CALLABLE
	virtual ~OVertex_PosShared() {};
	FPoint4* pPos;
	union
	{
		struct
		{
			FVector2 uv;
			FVector3 n;
			FVector3 tan;
			FVector3 vd;
			RGBColor c;
		};
		OVertexData vData;
	};
};