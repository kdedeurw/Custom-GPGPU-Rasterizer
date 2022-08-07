#pragma once
#include "GPUHelpers.h"

struct GPUTextureCompact
{
	cudaTextureObject_t dev_pTex; //cuda texture pointer address
	unsigned int* dev_TextureData; //pixel array
};

struct GPUTexture : public GPUTextureCompact //size == 6
{
	unsigned int w; //width
	unsigned int h; //height
	const static unsigned int bpp = 4; //bytes per pixel
};

struct GPUTextures
{
	GPUTexture Diff;
	GPUTexture Norm;
	GPUTexture Spec;
	GPUTexture Gloss;
};

struct GPUTexturesCompact //saves about 24 precious bytes
{
	GPUTextureCompact Diff;
	GPUTextureCompact Norm;
	GPUTextureCompact Spec;
	GPUTextureCompact Gloss;
	unsigned int w;
	unsigned int h;
	const static unsigned int bpp = 4;
};