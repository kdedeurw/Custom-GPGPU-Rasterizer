#include "PCH.h"

//TODO: not needed to include math headers, yet they are undefined/unrecognised by the compiler and does not compile
//https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INT.html#group__CUDA__MATH__INT
//https://docs.nvidia.com/cuda/pdf/CUDA_Math_API.pdf

BOTH_CALLABLE GPU_INLINE static
float GetMinElement(float val0, float val1, float val2)
{
	float min = val0;
	if (val1 < min)
		min = val1;
	if (val2 < min)
		min = val2;
	return min;
}

BOTH_CALLABLE GPU_INLINE static
float GetMaxElement(float val0, float val1, float val2)
{
	float max = val0;
	if (val1 > max)
		max = val1;
	if (val2 > max)
		max = val2;
	return max;
}

template <typename T>
BOTH_CALLABLE GPU_INLINE static
T ClampFast(T val, T min, T max)
{
	const T clamp = val < min ? min : val;
	return clamp > max ? max : clamp;
}

GPU_CALLABLE GPU_INLINE static
float EdgeFunction(const FPoint2& v, const FVector2& edge, const FPoint2& pixel);

GPU_CALLABLE GPU_INLINE static
bool IsPixelInTriangle(const FPoint2& v0, const FPoint2& v1, const FPoint2& v2, const FPoint2& pixel, float weights[3]);

GPU_CALLABLE static
bool IsAllXOutsideFrustum(const FPoint3& v0, const FPoint3& v1, const FPoint3& v2);

GPU_CALLABLE static
bool IsAllYOutsideFrustum(const FPoint3& v0, const FPoint3& v1, const FPoint3& v2);

GPU_CALLABLE static
bool IsAllZOutsideFrustum(const FPoint3& v0, const FPoint3& v1, const FPoint3& v2);

GPU_CALLABLE static
bool IsTriangleVisible(const FPoint3& v0, const FPoint3& v1, const FPoint3& v2);

GPU_CALLABLE static
bool IsVertexInFrustum(const FPoint3& NDC);

GPU_CALLABLE static
bool IsTriangleInFrustum(const FPoint3& v0, const FPoint3& v1, const FPoint3& v2);

BOTH_CALLABLE static
BoundingBox GetBoundingBox(const FPoint2& v0, const FPoint2& v1, const FPoint2& v2, const unsigned int width, const unsigned int height);

BOTH_CALLABLE GPU_INLINE static
void NDCToScreenSpace(FPoint2& v0, FPoint2& v1, FPoint2& v2, const unsigned int width, const unsigned int height);

GPU_CALLABLE GPU_INLINE static
OVertex GetNDCVertex(const IVertex& __restrict__ iVertex, const FMatrix4& wvpMat, const FMatrix4& worldMat, const FMatrix3& rotMat, const FPoint3& camPos);

GPU_CALLABLE GPU_INLINE static
float InterpolateElement(const float v0, const float v1, const float v2, const float weights[3], const float invDepths[3]);

GPU_CALLABLE GPU_INLINE static
FVector2 InterpolateVector(const FVector2& v0, const FVector2& v1, const FVector2& v2, const float weights[3], const float invDepths[3]);

GPU_CALLABLE GPU_INLINE static
FVector3 InterpolateVector(const FVector3& v0, const FVector3& v1, const FVector3& v2, const float weights[3], const float invDepths[3]);

GPU_CALLABLE GPU_INLINE static
RGBColor InterpolateColour(const RGBColor& v0, const RGBColor& v1, const RGBColor& v2, const float weights[3], const float invDepths[3]);

GPU_CALLABLE static
bool IsPixelInBoundingBox(const FPoint2& pixel, const BoundingBox& bb);

GPU_CALLABLE static
unsigned int GetStridedIdxByOffset(unsigned int globalDataIdx, unsigned int vertexStride, unsigned int valueStride, unsigned int offset = 0);

GPU_CALLABLE static
void PerformDepthTestAtomic(int* dev_DepthBuffer, int* dev_DepthMutexBuffer, const unsigned int pixelIdx, float zInterpolated, PixelShade* dev_PixelShadeBuffer, const PixelShade& pixelShade);

GPU_CALLABLE static
void RasterizePixel(const FPoint2& pixel, const OVertex& v0, const OVertex& v1, const OVertex& v2,
	int* dev_DepthBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const CUDATexturesCompact& textures);

GPU_CALLABLE static
void RasterizePixelAtomic(const FPoint2& pixel, const OVertex& v0, const OVertex& v1, const OVertex& v2,
	int* dev_DepthBuffer, int* dev_DepthMutexBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const CUDATexturesCompact& textures);

GPU_CALLABLE static
void RasterizeTriangle(const BoundingBox& bb, const OVertex& v0, const OVertex& v1, const OVertex& v2,
	int* dev_DepthMutexBuffer, int* dev_DepthBuffer, PixelShade* dev_PixelShadeBuffer, unsigned int width, const CUDATexturesCompact& textures);

GPU_CALLABLE GPU_INLINE static
RGBColor ShadePixel(const CUDATexturesCompact& textures, const FVector2& uv, const FVector3& n, const FVector3& tan, const FVector3& vd,
	SampleState sampleState, bool isFlipGreenChannel = false);

GPU_CALLABLE GPU_INLINE static
void MultiplyMatVec(const float* pMat, float* pVec, unsigned int matSize, unsigned int vecSize);

GPU_CALLABLE GPU_INLINE static
void CalculateOutputPosXYZ(const float* pMat, float* pVec);

GPU_CALLABLE GPU_INLINE static
void CalculateOutputPosXYZW(const float* pMat, float* pVec, float* pW);

//BINNING + TILING

GPU_CALLABLE static
BoundingBox GetBoundingBoxTiled(const FPoint2& v0, const FPoint2& v1, const FPoint2& v2,
	const unsigned int minX, const unsigned int minY, const unsigned int maxX, const unsigned int maxY);

GPU_CALLABLE static
bool IsOverlapping(const BoundingBox& rect0, const BoundingBox& rect1);