#include "PCH.h"
#include "Mesh.h"

Mesh::Mesh(std::vector<IVertex>& vertexBuffer, short stride, short type, 
	std::vector<unsigned int>& indexBuffer, PrimitiveTopology topology, const FPoint3& pos)
	: m_VertexType{ type }
	, m_VertexStride{ stride }
	, m_Topology{ topology }
	, m_VertexBuffer{}
	, m_IndexBuffer{}
	, m_Position{ reinterpret_cast<FPoint3&>(m_WorldSpace[3][0]) }
	, m_TextureIds{ -1, -1, -1, -1 }
	, m_WorldSpace{ FMatrix4::Identity() }
{
	m_VertexBuffer.swap(vertexBuffer);
	m_IndexBuffer.swap(indexBuffer);
	m_Position += reinterpret_cast<const FVector3&>(pos);
}

Mesh::~Mesh()
{
}

void Mesh::SetTextureIds(int diff, int norm, int spec, int gloss)
{
	m_TextureIds[0] = diff;
	m_TextureIds[1] = norm;
	m_TextureIds[2] = spec;
	m_TextureIds[3] = gloss;
}

void Mesh::SetTextureId(int id, TextureID texID)
{
	m_TextureIds[texID] = id;
}

unsigned int Mesh::GetTotalNumTriangles() const
{
	const unsigned int numIndices = GetNumIndices();
	unsigned int numTriangles{};
	switch (m_Topology)
	{
	case PrimitiveTopology::TriangleList:
		numTriangles += numIndices / 3;
		break;
	case PrimitiveTopology::TriangleStrip:
		numTriangles += numIndices - 2;
		break;
	}
	return numTriangles;
}