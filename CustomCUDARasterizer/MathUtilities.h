/*=============================================================================*/
// Authors: Matthieu Delaere, Thomas Goussaert
/*=============================================================================*/
// EMathUtilities.h: Utility class containing a bunch of commonly used functionality (not custom-type specific)
/*=============================================================================*/
#pragma once

//Standard C++ includes
#include <cstdint>
#include <cstdlib>
#include <cfloat>
//Limits: possible because of experimental branch compile command (--expt-relaxed-constexpr)
#include <limits>
#include <type_traits>

#include "GPUHelpers.h"

/* --- CONSTANTS --- */
#define PI		3.14159265358979323846
#define PI_DIV2	1.57079632679489661923
#define PI_DIV4	0.785398163397448309616
#define PI_2	6.283185307179586476925
#define PI_4	12.56637061435917295385

#define TO_DEGREES (180.0 / PI)
#define TO_RADIANS (E_PI /180.0)

/* --- FUNCTIONS --- */
template<typename T>
//typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
bool AreEqual(T a, T b, uint16_t ulp = 2) //Only works when non-integer types used
{
	// Source: cpp-reference, the machine epsilon has to be scaled to the magnitude of the values used
	// and multiplied by the desired precision in ULPs (units in the last place)
	return std::abs(a - b) <= std::numeric_limits<T>::epsilon() * std::abs(a + b) * ulp
		// unless the result is subnormal
		|| std::abs(a - b) < std::numeric_limits<T>::min();
}

/*! An accurate inverse square root*/
template<typename T>
BOTH_CALLABLE inline T InvSqrt(T a)
{
	return static_cast<T>(1.0 / sqrt(a));
}

/*! An fast inverse square root, not fully accurate. Implementation based on Quake III Arena*/
/*! Reference: https://betterexplained.com/articles/understanding-quakes-fast-inverse-square-root/ */
template<typename T>
BOTH_CALLABLE const T InvSqrtFst(T f)
{
	const T xHalf = 0.5f * f;
	int32_t i = *reinterpret_cast<int32_t*>(&f);
	i = 0x5f3759df - (i >> 1);
	f = *reinterpret_cast<T*>(&i);
	f = f * (1.5f - xHalf * f * f);
	return f;
}

template<typename T>
BOTH_CALLABLE void Swap(T&& a, T&& b)
{
	T temp = a;
	b = std::move(a);
	a = std::move(temp);
}

/*! Function to square a number -- Using experimental constexpr in CUDA! */
template<typename T, typename = std::enable_if<std::is_pod<T>::value>>
BOTH_CALLABLE constexpr T Square(T v)
{
	return v * v;
}
/*! Function to convert degrees to radians -- Using experimental constexpr in CUDA! */
template<typename T, typename = std::enable_if<std::is_pod<T>::value>>
BOTH_CALLABLE constexpr T ToRadians(const T angle)
{
	return angle * (static_cast<T>(PI) / static_cast<T>(180.f));
}
/*! Template function to clamp between a minimum and a maximum value -> in STD since c++17 */
/*! Using experimental constexpr in CUDA! */
template<typename T>
BOTH_CALLABLE constexpr T Clamp(const T a, T min, T max)
{
	if (a < min)
		return min;

	if (a > max)
		return max;

	return a;
}

//--Own Math functions
template<typename T>
BOTH_CALLABLE constexpr T Min(const T a, T b)
{
	if (a < b)
		return a;
	return b;
}

template<typename T>
BOTH_CALLABLE constexpr T Max(const T a, T b)
{
	if (a < b)
		return b;
	return a;
}
//--

/*! Set Random Seed */
inline void SetRandomSeed(const int32_t seed)
{
	srand(seed);
}

/*! Random Integer */
inline int32_t RandomInt32(int32_t max = 1)
{
	return rand() % max;
}

/*! Random Float */
inline float RandomFloat(float max = 1.f)
{
	return max * (float(rand()) / RAND_MAX);
}

/*! Random Binomial Float */
inline float RandomBinomial(float max = 1.f)
{
	return RandomFloat(max) - RandomFloat(max);
}

/*! Linear Interpolation */
/*inline float Lerp(float v0, float v1, float t)
{ return (1 - t) * v0 + t * v1;	}*/
template<typename T>
BOTH_CALLABLE inline T Lerp(T v0, T v1, float t)
{
	return (1 - t) * v0 + t * v1;
}

/*! Smooth Step */
BOTH_CALLABLE inline float SmoothStep(float edge0, float edge1, float x)
{
	// Scale, bias and saturate x to 0..1 range
	x = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	// Evaluate polynomial
	return x * x * (3 - 2 * x);
}

/*! Sign Function*/
template <typename T>
BOTH_CALLABLE int32_t Sign(T val)
{ return (T(0) < val) - (val < T(0)); }

/*! Remap Function*/
template<typename T>
BOTH_CALLABLE T Remap(T val, T min, T max)
{ return (val - min) / (max - min); }