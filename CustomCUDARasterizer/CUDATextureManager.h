#pragma once
#include <vector>
#include <string>

class CUDATexture;
class CUDATextureManager final
{
public:
	CUDATextureManager();
	~CUDATextureManager();

	CUDATexture* GetCUDATexture(int id) const;
	int AddCUDATexture(CUDATexture* pTex);
	void RemoveCUDATexture(int id);

private:
	unsigned int m_CurrentTextureId;
	std::vector<CUDATexture*> m_pCUDATextures;

};