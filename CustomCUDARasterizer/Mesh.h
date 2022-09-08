#pragma once
#include "Vertex.h"
#include "PrimitiveTopology.h"
#include "VertexType.h"

class Mesh
{
public:
	enum TextureID
	{
		Diffuse,
		Normal,
		Specular,
		Glossiness,
	};
	Mesh() = default;
	Mesh(IVertex* pVertexBuffer, unsigned int numVertices, short stride, short type, unsigned int* pIndexBuffer, unsigned int numIndices,
		PrimitiveTopology topology, const FPoint3& pos = {});
	virtual ~Mesh();

	FPoint3& GetPosition() { return m_Position; }
	FMatrix4& GetWorld() { return m_WorldSpace; }
	const FPoint3& GetPositionConst() const { return m_Position; }
	const FMatrix4& GetWorldConst() const { return m_WorldSpace; }
	FMatrix3 GetRotationMatrix() const { return (FMatrix3)m_WorldSpace; }

	short GetVertexType() const { return m_VertexType; }
	short GetVertexStride() const { return m_VertexStride; }
	PrimitiveTopology GetTopology() const { return m_Topology; }
	unsigned int GetNumVertices() const { return m_NumVertices; }
	unsigned int GetNumIndices() const { return m_NumIndices; }
	IVertex* GetVertexBuffer() const { return m_pVertexBuffer; }
	unsigned int* GetIndexBuffer() const { return m_pIndexBuffer; }
	const int* GetTextureIds() const { return m_TextureIds; }
	void SetTextureIds(int diff, int norm = -1, int spec = -1, int gloss = -1);
	void SetTextureId(int id, TextureID texID = TextureID::Diffuse);

protected:
	short m_VertexType;
	short m_VertexStride;
	PrimitiveTopology m_Topology;
	unsigned int m_NumVertices;
	unsigned int m_NumIndices;
	FPoint3& m_Position;
	IVertex* m_pVertexBuffer;
	unsigned int* m_pIndexBuffer;
	int m_TextureIds[4];
	FMatrix4 m_WorldSpace;
};