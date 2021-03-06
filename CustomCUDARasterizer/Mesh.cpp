#include "PCH.h"
#include "Mesh.h"
#include "Texture.h"

Mesh::Mesh(const std::vector<IVertex>& vertices, const std::vector<unsigned int>& indexes, 
	PrimitiveTopology pT, float rotSpeed, const FPoint3& position)
	: m_VertexBuffer{vertices}
	, m_IndexBuffer{indexes}
	, m_TexturePaths{}
	, m_Topology{pT}
	, m_pTextures{}
	, m_WorldSpace{ FMatrix4::Identity() }
	, m_RotateSpeed{ rotSpeed }
{
	m_WorldSpace.data[0][0] += position.x;
	m_WorldSpace.data[1][1] += position.y;
	m_WorldSpace.data[2][2] += position.z;
}

Mesh::~Mesh()
{
	if (m_pTextures.pDiff)
	{
		delete m_pTextures.pDiff;
		m_pTextures.pDiff = nullptr;
	}
	if (m_pTextures.pNorm)
	{
		delete m_pTextures.pNorm;
		m_pTextures.pNorm = nullptr;
	}
	if (m_pTextures.pSpec)
	{
		delete m_pTextures.pSpec;
		m_pTextures.pSpec = nullptr;
	}
	if (m_pTextures.pGloss)
	{
		delete m_pTextures.pGloss;
		m_pTextures.pGloss = nullptr;
	}
}

void Mesh::Update(float elapsedSec)
{
	m_WorldSpace *= (FMatrix4)MakeRotationY(m_RotateSpeed * elapsedSec);
}

void Mesh::LoadTextures(const std::string texPaths[4])
{
	m_TexturePaths[0] = texPaths[0];
	m_TexturePaths[1] = texPaths[1];
	m_TexturePaths[2] = texPaths[2];
	m_TexturePaths[3] = texPaths[3];

	if (!m_pTextures.pDiff && texPaths[0] != "")
		m_pTextures.pDiff = new Texture{ texPaths[0].c_str() };

	if (!m_pTextures.pNorm && texPaths[1] != "")
		m_pTextures.pNorm = new Texture{ texPaths[1].c_str() };

	if (!m_pTextures.pSpec && texPaths[2] != "")
		m_pTextures.pSpec = new Texture{ texPaths[2].c_str() };

	if (!m_pTextures.pGloss && texPaths[3] != "")
		m_pTextures.pGloss = new Texture{ texPaths[3].c_str() };
}