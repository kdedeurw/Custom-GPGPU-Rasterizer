#pragma once
#include "GPUHelpers.h"

struct GPUTexture
{
	cudaTextureObject_t dev_pTex; //cuda texture pointer address
	unsigned int w; //height
	unsigned int h; //width
	unsigned int* dev_TextureData;
	const static unsigned int bpp = 4; //bytes per pixel
};

struct GPUTextures
{
	GPUTexture Diff;
	GPUTexture Norm;
	GPUTexture Spec;
	GPUTexture Gloss;
};