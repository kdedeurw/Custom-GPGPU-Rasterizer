#include "PCH.h"
#include "Mesh.h"

Mesh::Mesh(const std::vector<IVertex>& vertexBuffer, short stride, short type, 
	const std::vector<unsigned int>& indexBuffer, PrimitiveTopology topology, const FPoint3& pos)
	: Mesh{ nullptr, (unsigned int)vertexBuffer.size(), stride, type, nullptr, (unsigned int)indexBuffer.size(), topology, pos }
{
	m_pVertexBuffer = new IVertex[m_NumVertices];
	memcpy(m_pVertexBuffer, vertexBuffer.data(), sizeof(IVertex) * m_NumVertices);
	m_pIndexBuffer = new unsigned int[m_NumIndices];
	memcpy(m_pIndexBuffer, indexBuffer.data(), sizeof(unsigned int) * m_NumIndices);
	m_Position += reinterpret_cast<const FVector3&>(pos);
}

Mesh::Mesh(IVertex* pVertexBuffer, unsigned int numVertices, short stride, short type, unsigned int* pIndexBuffer, unsigned int numIndices,
	PrimitiveTopology topology, const FPoint3& pos)
	: m_VertexType{ type }
	, m_VertexStride{ stride }
	, m_Topology{ topology }
	, m_NumVertices{ numVertices }
	, m_NumIndices{ numIndices }
	, m_Position{ reinterpret_cast<FPoint3&>(m_WorldSpace[3][0]) }
	, m_pVertexBuffer{ pVertexBuffer }
	, m_pIndexBuffer{ pIndexBuffer }
	, m_TextureIds{ -1, -1, -1, -1 }
	, m_WorldSpace{ FMatrix4::Identity() }
{
	m_Position += reinterpret_cast<const FVector3&>(pos);
}

Mesh::~Mesh()
{
	delete[] m_pVertexBuffer;
	delete[] m_pIndexBuffer;
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