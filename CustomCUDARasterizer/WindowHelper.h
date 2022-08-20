#pragma once
#include "Resolution.h"

struct SDL_Window;
struct SDL_Surface;
struct WindowHelper
{
	SDL_Window* pWindow = nullptr;
	SDL_Surface* pFrontBuffer = nullptr;
	SDL_Surface* pBackBuffer = nullptr;
	unsigned int* pBackBufferPixels = nullptr;
	Resolution Resolution;
};