#include "PCH.h"
#include "Texture.h"
#include "MathUtilities.h"
#include <SDL_image.h>
#include <SDL_surface.h>
#include <iostream>

Texture::Texture(const char* file)
	: m_pSurface{ IMG_Load(file) }
{
	if (!m_pSurface)
		std::cout << "\n!Texture not loaded correctly!\n";
}

Texture::~Texture()
{
	if (m_pSurface)
		SDL_FreeSurface(m_pSurface);
	m_pSurface = nullptr;
}