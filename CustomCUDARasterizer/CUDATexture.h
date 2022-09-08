#pragma once
#include "GPUHelpers.h"

class CUDATexture
{
public:
	CUDATexture();
	CUDATexture(const std::string& path);
	virtual ~CUDATexture() noexcept;

	bool IsAllocated() const { return m_Width != 0 && m_Height != 0; }
	virtual bool Create(const char* path);
	virtual void Destroy();

	unsigned int GetBytesPerPixel() const { return m_BytesPerPixel; }
	unsigned int GetWidth() const { return m_Width; }
	unsigned int GetHeight() const { return m_Height; }

	cudaTextureObject_t GetTextureObject() const { return m_Dev_pTex; }
	unsigned int* GetTextureData() const { return m_Dev_TextureData; }

protected:
	unsigned int m_BytesPerPixel; //bytes per pixel
	unsigned int m_Width; //width
	unsigned int m_Height; //height
	cudaTextureObject_t m_Dev_pTex; //cuda texture pointer address
	unsigned int* m_Dev_TextureData; //pixel array
};