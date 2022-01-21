#pragma once

struct SDL_Window;
struct SDL_Surface;
struct WindowHelper
{
	SDL_Window* pWindow = nullptr;
	SDL_Surface* pFrontBuffer = nullptr;
	SDL_Surface* pBackBuffer = nullptr;
	unsigned int* pBackBufferPixels = nullptr;
	float* pDepthBuffer;
	unsigned int Width = 0;
	unsigned int Height = 0;
};