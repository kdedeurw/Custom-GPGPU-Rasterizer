#include "PCH.h"
#include "CUDAMesh.h"

CUDAMesh::CUDAMesh(const Mesh* pMesh)
	: m_VertexType{ pMesh->GetVertexType() }
	, m_VertexStride{ pMesh->GetVertexStride() }
	, m_Topology{ pMesh->GetTopology() }
	, m_NumVertices{ pMesh->GetNumVertices() }
	, m_NumIndices{ pMesh->GetNumIndices() }
	, m_Dev_IVertexBuffer{}
	, m_Dev_IndexBuffer{}
	, m_Dev_OVertexBuffer{}
	, m_Dev_TriangleBuffer{}
	, m_Position{ reinterpret_cast<FPoint3&>(m_WorldSpace[3][0]) }
	, m_Textures{}
	, m_WorldSpace{ pMesh->GetWorldConst() }
{
	//m_Position += reinterpret_cast<const FVector3&>(pMesh->GetPositionConst());
	Allocate(pMesh->GetVertexBuffer().data(), pMesh->GetIndexBuffer().data());
};

CUDAMesh::~CUDAMesh()
{
	Free();
}

unsigned int CUDAMesh::GetTotalNumTriangles() const
{
	unsigned int numTriangles{};
	switch (m_Topology)
	{
	case PrimitiveTopology::TriangleList:
		numTriangles += m_NumIndices / 3;
		break;
	case PrimitiveTopology::TriangleStrip:
		numTriangles += m_NumIndices - 2;
		break;
	}
	return numTriangles;
}

void CUDAMesh::Allocate(const IVertex* pVertexBuffer, const unsigned int* pIndexBuffer)
{
	const unsigned int numTriangles = GetTotalNumTriangles();

	//Allocate Input Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_IVertexBuffer, m_NumVertices * sizeof(IVertex)));
	//Allocate Index Buffer
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_IndexBuffer, m_NumIndices * sizeof(unsigned int)));
	//Allocate Ouput Vertex Buffer
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_OVertexBuffer, m_NumVertices * sizeof(OVertex)));
	//Allocate device memory for entire range of triangles
	CheckErrorCuda(cudaMalloc((void**)&m_Dev_TriangleBuffer, numTriangles * sizeof(TriangleIdx)));

	//Copy Input Vertex Buffer
	CheckErrorCuda(cudaMemcpy(m_Dev_IVertexBuffer, pVertexBuffer, m_NumVertices * sizeof(IVertex), cudaMemcpyHostToDevice));
	//Copy Index Buffer
	CheckErrorCuda(cudaMemcpy(m_Dev_IndexBuffer, pIndexBuffer, m_NumIndices * sizeof(unsigned int), cudaMemcpyHostToDevice));
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