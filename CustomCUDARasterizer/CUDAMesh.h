#pragma once

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
class CUDAMesh
{
public:
	CUDAMesh() = delete;
	CUDAMesh(const CUDAMesh& other) = delete;
	CUDAMesh(CUDAMesh&& other) = delete;
	CUDAMesh& operator=(const CUDAMesh& other) = delete;
	CUDAMesh& operator=(CUDAMesh&& other) = delete;

	explicit CUDAMesh(const unsigned int idx, const Mesh* pMesh);
	virtual ~CUDAMesh();

	unsigned int GetIdx() const { return m_Idx; }
	unsigned int GetTotalNumTriangles() const;
	unsigned int& GetVisibleNumTriangles() { return m_VisibleNumTriangles; }
	unsigned int GetVisibleNumTrianglesConst() const { return m_VisibleNumTriangles; }
	const Mesh* GetMesh() const { return m_pMesh; }
	void SetTextures(const CUDATexturesCompact& textures) { m_Textures = textures; }
	const CUDATexturesCompact& GetTextures() const { return m_Textures; }

	IVertex* GetDevIVertexBuffer() const { return m_Dev_IVertexBuffer; }
	unsigned int* GetDevIndexBuffer() const { return m_Dev_IndexBuffer; }
	OVertex* GetDevOVertexBuffer() const { return m_Dev_OVertexBuffer; }
	TriangleIdx* GetDevTriangleBuffer() const { return m_Dev_TriangleBuffer; }

protected:
	unsigned int m_Idx;
	unsigned int m_VisibleNumTriangles;
	const Mesh* m_pMesh;
	IVertex* m_Dev_IVertexBuffer;
	unsigned int* m_Dev_IndexBuffer;
	OVertex* m_Dev_OVertexBuffer;
	TriangleIdx* m_Dev_TriangleBuffer;
	CUDATexturesCompact m_Textures;

	virtual void Allocate();
	virtual void Free();
};