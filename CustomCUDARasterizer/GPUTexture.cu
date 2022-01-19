#include "PCH.h"
#include "GPUTexture.cuh"
#include "MathUtilities.h"
#include <SDL_image.h>
#include <SDL_surface.h>
#include <iostream>

CPU_CALLABLE GPUTexture::GPUTexture(const char* file)
	: m_pSurface{ IMG_Load(file) }
{
	if (!m_pSurface)
		std::cout << "\n!Texture not loaded correctly!\n";
}

CPU_CALLABLE GPUTexture::~GPUTexture()
{
	if (m_pSurface)
		SDL_FreeSurface(m_pSurface);
	m_pSurface = nullptr;
}