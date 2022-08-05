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
	explicit Mesh(const std::vector<IVertex>& vertices, const std::vector<unsigned int>& indexes, short vType,
		PrimitiveTopology pT = PrimitiveTopology::TriangleList, const FPoint3& position = FPoint3{});
	virtual ~Mesh();

	const std::vector<IVertex>& GetVertices() const { return m_VertexBuffer; };
	const std::vector<unsigned int>& GetIndexes() const { return m_IndexBuffer; };
	PrimitiveTopology GetTopology() const { return m_Topology; };
	short GetVertexType() const { return m_VertexType; };
	const Textures& GetTextures() const { return m_pTextures; };
	const FMatrix4& GetWorldMatrix() const { return m_WorldSpace; };

	void Update(float elapsedSec);
	void LoadTextures(const std::string texPaths[4]);
	const std::string* GetTexPaths() const { return m_TexturePaths; };

private:
	const PrimitiveTopology m_Topology;
	const short m_VertexType;
	Textures m_pTextures;
	FMatrix4 m_WorldSpace;
	std::string m_TexturePaths[4];
	const std::vector<IVertex> m_VertexBuffer;
	const std::vector<unsigned int> m_IndexBuffer;
};