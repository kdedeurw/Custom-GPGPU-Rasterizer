#include "PCH.h"
#include "CUDATextureManager.h"
#include "CUDATexture.h"

CUDATextureManager::CUDATextureManager()
	: m_CurrentTextureId{}
	, m_pCUDATextures{}
{
	m_pCUDATextures.reserve(4);
}

CUDATextureManager::~CUDATextureManager()
{
	for (CUDATexture* pCUDATexture : m_pCUDATextures)
	{
		delete pCUDATexture;
	}
	m_pCUDATextures.clear();
}

CUDATexture* CUDATextureManager::GetCUDATexture(int id) const
{
	if (id >= 0 && m_CurrentTextureId > id)
		return m_pCUDATextures[id];

	return nullptr;
}

int CUDATextureManager::AddCUDATexture(CUDATexture* pTex)
{
	if (!pTex)
		throw std::runtime_error{ "CUDATextureManager::AddCUDATexture > pTex is invalid" };

	m_pCUDATextures.push_back(pTex);

	return m_CurrentTextureId++;
}

void CUDATextureManager::RemoveCUDATexture(int id)
{
	if (m_CurrentTextureId < id && id < 0)
		throw std::runtime_error{ "CUDATextureManager::RemoveCUDATexture > id is invalid" };

	CUDATexture* pTex = GetCUDATexture(id);
	if (!pTex)
		return;

	const auto it = std::find(m_pCUDATextures.cbegin(), m_pCUDATextures.cend(), pTex);
	if (it == m_pCUDATextures.cend())
		return;

	m_pCUDATextures.erase(it);

	delete pTex;
}