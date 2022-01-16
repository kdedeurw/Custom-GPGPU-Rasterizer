#pragma once
#include "EMath.h"
#include "EMathUtilities.h"
#include "ERGBColor.h"
using namespace Elite;

enum class SampleState
{
	Point,
	Linear,
};

struct SDL_Surface;
class Texture
{
public:
	explicit Texture(const char* file);
	~Texture();

	RGBColor Sample(const FVector2& uv, SampleState state = SampleState::Point) const;

private:
	SDL_Surface* m_pSurface;

	RGBColor SamplePoint(const FVector2& uv) const;
	RGBColor SampleLinear(const FVector2& uv) const;
};