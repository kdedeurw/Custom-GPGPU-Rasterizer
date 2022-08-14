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
	explicit Mesh(std::vector<IVertex>& vertexBuffer, std::vector<unsigned int>& indexBuffer,
		PrimitiveTopology pT = PrimitiveTopology::TriangleList, const FPoint3& position = FPoint3{});
	virtual ~Mesh();

	const std::vector<IVertex>& GetVertexBuffer() const { return m_VertexBuffer; };
	const std::vector<unsigned int>& GetIndexBuffer() const { return m_IndexBuffer; };
	PrimitiveTopology GetTopology() const { return m_Topology; };
	short GetVertexStride() const { return sizeof(IVertex); };
	unsigned int GetVertexAmount() const { return m_VertexBuffer.size(); };
	unsigned int GetIndexAmount() const { return m_IndexBuffer.size(); };
	const Textures& GetTextures() const { return m_pTextures; };
	const FMatrix4& GetWorldMatrix() const { return m_WorldSpace; };
	FMatrix3 GetRotationMatrix() const { return (FMatrix3)m_WorldSpace; };

	virtual void Update(float elapsedSec);
	void LoadTextures(const std::string texPaths[4]);
	const std::string* GetTexPaths() const { return m_TexturePaths; };

protected:
	const PrimitiveTopology m_Topology;
	std::vector<IVertex> m_VertexBuffer;
	std::vector<unsigned int> m_IndexBuffer;
	Textures m_pTextures;
	FMatrix4 m_WorldSpace;
	std::string m_TexturePaths[4];
};