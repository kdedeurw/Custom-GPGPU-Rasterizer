#pragma once

#include "Vertex.h"
#include <vector>
#include <string>

class Texture;

class Mesh
{
public:
	enum class PrimitiveTopology
	{
		TriangleList,
		TriangleStrip
	};
	struct Textures
	{
		Texture* pDiff{};
		Texture* pNorm{};
		Texture* pSpec{};
		Texture* pGloss{};
	};

	explicit Mesh(const std::vector<IVertex>& vertices, const std::vector<int>& indexes, const std::string texPaths[4], PrimitiveTopology pT = PrimitiveTopology::TriangleList, float rotSpeed = 0.f, const FPoint3& position = FPoint3{});
	virtual ~Mesh();

	const std::vector<IVertex>& GetVertices() const;
	const std::vector<int>& GetIndexes() const;
	const PrimitiveTopology GetTopology() const;
	const Textures& GetTextures() const;
	const FMatrix4& GetWorldMatrix() const;

	void Update(float elapsedSec);

private:
	float m_RotateSpeed;
	const PrimitiveTopology m_Topology;
	Textures m_pTextures;
	FMatrix4 m_WorldSpace;
	const std::vector<IVertex> m_VertexBuffer;
	const std::vector<int> m_IndexBuffer;
};

