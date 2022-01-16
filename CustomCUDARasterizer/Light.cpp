#include "PCH.h"
#include "Light.h"

Light::Light(const RGBColor& colour, float intensity)
	: m_Colour{colour}
	, m_Intensity{intensity}
{}