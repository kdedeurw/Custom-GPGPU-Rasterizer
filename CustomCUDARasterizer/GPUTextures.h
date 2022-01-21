#pragma once

struct GPUTexture
{
	const unsigned int* pixels;
	const unsigned int w;
	const unsigned int h;
	//TODO: format? (SDL_Format)
};

struct GPUTextures
{
	GPUTexture* pDiff{};
	GPUTexture* pNorm{};
	GPUTexture* pSpec{};
	GPUTexture* pGloss{};
};