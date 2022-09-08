#pragma once
#include "Math.h"
#include "RGBColor.h"
#include "CUDATexture.h"

struct CUDATextureCompact
{
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
	unsigned int colour;
	float zInterpolated;
	float wInterpolated;
	FVector2 uv;
	FVector3 n;
	FVector3 tan;
	FVector3 vd;
	CUDATexturesCompact textures;
};