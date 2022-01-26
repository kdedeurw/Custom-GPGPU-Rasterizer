#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

#include "RGBColor.h"
#include "Math.h"
#include "GPUTextures.h"
#include "RGBRaw.h"

namespace GPUTextureSampler
{
	GPU_CALLABLE GPU_INLINE RGBColor Sample(const GPUTexture& gpuTexture, const FVector2& uv);
}

GPU_CALLABLE GPU_INLINE
RGBColor GPUTextureSampler::Sample(const GPUTexture& gpuTexture, const FVector2& uv)
{
	//// Transform coordinates
	float u{ uv.x };
	float v{ uv.y };
	u *= gpuTexture.w;
	v *= gpuTexture.h;
	u += 0.5f;
	v += 0.5f;
	//u -= 0.5f;
	//v -= 0.5f;
	//float tu = u * cosf(theta) – v * sinf(theta) + 0.5f;
	//float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

	unsigned int sample = tex1Dfetch<unsigned int>(gpuTexture.dev_pTex, u + v * gpuTexture.w);
	const RGBA rgba{ sample };
	return RGBColor{ rgba.values.b / 255.f, rgba.values.g / 255.f, rgba.values.r / 255.f };
}