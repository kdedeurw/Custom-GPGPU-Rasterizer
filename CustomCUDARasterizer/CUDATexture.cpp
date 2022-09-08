#include "PCH.h"
#include "CUDATexture.h"
#include <SDL_image.h>

CUDATexture::CUDATexture()
	: m_BytesPerPixel{}
	, m_Width{}
	, m_Height{}
	, m_Dev_pTex{}
	, m_Dev_TextureData{}
{}

CUDATexture::CUDATexture(const std::string& path)
	: CUDATexture{}
{
	Create(path.c_str());
}

CUDATexture::~CUDATexture()
{
	Destroy();
}

bool CUDATexture::Create(const char* path)
{
	Destroy();

	SDL_Surface* pSurface = IMG_Load(path);
	if (!pSurface)
		return false;

	const unsigned int width = pSurface->w;
	const unsigned int height = pSurface->h;
	const unsigned int* pixels = (unsigned int*)pSurface->pixels;
	const unsigned int bpp = pSurface->format->BytesPerPixel;
	//const unsigned int sizeInBytes = width * height * bpp;
	unsigned int* dev_TexData;

	//copy texture data to device
	size_t pitch{};
	CheckErrorCuda(cudaMallocPitch((void**)&dev_TexData, &pitch, width * bpp, height)); //2D array
	CheckErrorCuda(cudaMemcpy2D(dev_TexData, pitch, pixels, pitch, width * bpp, height, cudaMemcpyHostToDevice));

	//cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<unsigned int>();

	cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = dev_TexData;
	resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.pitch2D.desc.x = pSurface->format->BitsPerPixel;
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;

	cudaTextureDesc texDesc{};
	texDesc.normalizedCoords = true; //able to sample texture with normalized uv coordinates
	texDesc.filterMode = cudaFilterModePoint; //linear only supports float (and double) type
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t dev_TextureObject{};
	CheckErrorCuda(cudaCreateTextureObject(&dev_TextureObject, &resDesc, &texDesc, nullptr));

	m_Dev_pTex = dev_TextureObject;
	m_Width = width;
	m_Height = height;
	m_Dev_TextureData = dev_TexData;
	m_BytesPerPixel = bpp;

	//free data
	SDL_FreeSurface(pSurface);

	return true;
}

void CUDATexture::Destroy()
{
	CheckErrorCuda(cudaDestroyTextureObject(m_Dev_pTex));
	CheckErrorCuda(cudaFree(m_Dev_TextureData));
	m_Dev_pTex = (cudaTextureObject_t)0;
	m_Dev_TextureData = nullptr;
	m_Width = m_Height = 0;
	m_BytesPerPixel = 0;
}