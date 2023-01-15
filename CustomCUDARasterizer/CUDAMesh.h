#pragma once
#include "CUDAStructs.h"

struct IVertex;
struct OVertex;
struct TriangleIdx;
struct CUDAMeshBuffers
{
	IVertex* dev_IVertexBuffer;
	OVertex* dev_OVertexBuffer;
	unsigned int* dev_IndexBuffer;
	TriangleIdx* dev_TriangleBuffer;
};

class Mesh;
struct CUDATexturesCompact;
enum class PrimitiveTopology : unsigned char;
class CUDAMesh
{
public:
	CUDAMesh() = delete;
	CUDAMesh(const CUDAMesh& other) = delete;
	CUDAMesh(CUDAMesh&& other) = delete;
	CUDAMesh& operator=(const CUDAMesh& other) = delete;
	CUDAMesh& operator=(CUDAMesh&& other) = delete;

	explicit CUDAMesh(const Mesh* pMesh);
	virtual ~CUDAMesh();

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
	unsigned int GetTotalNumTriangles() const;
	void SetTextures(const CUDATexturesCompact& textures) { m_Textures = textures; }
	const CUDATexturesCompact& GetTextures() const { return m_Textures; }

	IVertex* GetDevIVertexBuffer() const { return m_Dev_IVertexBuffer; }
	unsigned int* GetDevIndexBuffer() const { return m_Dev_IndexBuffer; }
	OVertex* GetDevOVertexBuffer() const { return m_Dev_OVertexBuffer; }
	TriangleIdx* GetDevTriangleBuffer() const { return m_Dev_TriangleBuffer; }

protected:
	short m_VertexType;
	short m_VertexStride;
	PrimitiveTopology m_Topology;
	unsigned int m_NumVertices;
	unsigned int m_NumIndices;
	IVertex* m_Dev_IVertexBuffer;
	unsigned int* m_Dev_IndexBuffer;
	OVertex* m_Dev_OVertexBuffer;
	TriangleIdx* m_Dev_TriangleBuffer;
	FPoint3& m_Position;
	CUDATexturesCompact m_Textures;
	FMatrix4 m_WorldSpace;

	virtual void Allocate(IVertex* pVertexBuffer, unsigned int* pIndexBuffer);
	virtual void Free();
};