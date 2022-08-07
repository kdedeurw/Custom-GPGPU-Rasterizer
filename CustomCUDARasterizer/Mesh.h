#pragma once

#include "Vertex.h"
#include <vector>
#include <string>
#include "PrimitiveTopology.h"
#include "VertexType.h"
#include "Textures.h"

class Mesh
{
public:
	explicit Mesh(float* pVertices, unsigned int vertexAmount, short stride, unsigned int* pIndexes, unsigned int indexAmount,
		short vType, PrimitiveTopology pT = PrimitiveTopology::TriangleList, const FPoint3& position = FPoint3{});
	virtual ~Mesh();

	float* GetVertices() const { return m_pVertexBuffer; };
	unsigned int* GetIndexes() const { return m_pIndexBuffer; };
	PrimitiveTopology GetTopology() const { return m_Topology; };
	short GetVertexType() const { return m_VertexType; };
	short GetVertexStride() const { return m_VertexStride; };
	unsigned int GetVertexAmount() const { return m_VertexAmount; };
	unsigned int GetIndexAmount() const { return m_IndexAmount; };
	const Textures& GetTextures() const { return m_pTextures; };
	const FMatrix4& GetWorldMatrix() const { return m_WorldSpace; };

	virtual void Update(float elapsedSec) {};
	void LoadTextures(const std::string texPaths[4]);
	const std::string* GetTexPaths() const { return m_TexturePaths; };

	bool IsPoint4Pos() const { return m_VertexType | (short)VertexType::Pos4; }

protected:
	const PrimitiveTopology m_Topology;
	const unsigned char m_VertexStride;
	const short m_VertexType;
	float* m_pVertexBuffer;
	unsigned int* m_pIndexBuffer;
	unsigned int m_VertexAmount;
	unsigned int m_IndexAmount;
	Textures m_pTextures;
	FMatrix4 m_WorldSpace;
	std::string m_TexturePaths[4];
};