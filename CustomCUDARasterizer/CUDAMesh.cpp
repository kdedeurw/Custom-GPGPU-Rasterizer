#include "PCH.h"
#include "CUDAMesh.h"

CUDAMesh::CUDAMesh(const unsigned int idx, const Mesh* pMesh)
	: m_Idx{ idx }
	, m_TotalNumTriangles{}
	, m_VisibleNumTriangles{}
	, m_pMesh{ pMesh }
	, m_Dev_IVertexBuffer{}
	, m_Dev_IndexBuffer{}
	, m_Dev_OVertexBuffer{}
	, m_Dev_TriangleBuffer{}
	, m_Textures{}
{
	Allocate();
};

CUDAMesh::~CUDAMesh()
{
	Free();
}

void CUDAMesh::Allocate()
{
	const PrimitiveTopology topology = m_pMesh->GetTopology();
	const unsigned int numVertices = m_pMesh->GetNumVertices();
	const unsigned int numIndices = m_pMesh->GetNumIndices();
	const IVertex* pVertexBuffer = m_pMesh->GetVertexBuffer();
	const unsigned int* pIndexBuffer = m_pMesh->GetIndexBuffer();

	unsigned int numTriangles{};
	switch (topology)
	{
	case PrimitiveTopology::TriangleList:
		numTriangles += numIndices / 3;
		break;
	case PrimitiveTopology::TriangleStrip:
		numTriangles += numIndices - 2;
		break;
	}
	m_TotalNumTriangles = numTriangles;

	//Allocate Input Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_IVertexBuffer, numVertices * sizeof(IVertex)));
	//Allocate Index Buffer
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_IndexBuffer, numIndices * sizeof(unsigned int)));
	//Allocate Ouput Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_OVertexBuffer, numVertices * sizeof(OVertex)));
	//Allocate device memory for entire range of triangles
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_TriangleBuffer, numTriangles * sizeof(TriangleIdx)));

	//Copy Input Vertex Buffer
	CheckErrorCuda(cudaMemcpy(m_Dev_IVertexBuffer, pVertexBuffer, numVertices * sizeof(IVertex), cudaMemcpyHostToDevice));
	//Copy Index Buffer
	CheckErrorCuda(cudaMemcpy(m_Dev_IndexBuffer, pIndexBuffer, numIndices * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void CUDAMesh::Free()
{
	CheckErrorCuda(cudaFree(m_Dev_IVertexBuffer));
	m_Dev_IVertexBuffer = nullptr;
	CheckErrorCuda(cudaFree(m_Dev_IndexBuffer));
	m_Dev_IndexBuffer = nullptr;
	CheckErrorCuda(cudaFree(m_Dev_OVertexBuffer));
	m_Dev_OVertexBuffer = nullptr;
	CheckErrorCuda(cudaFree(m_Dev_TriangleBuffer));
	m_Dev_TriangleBuffer = nullptr;
}