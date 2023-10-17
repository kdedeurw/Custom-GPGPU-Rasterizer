#include "PCH.h"
#include "CUDASceneGraph.h"
#include <iostream>
#include "SceneGraph.h"
#include "Mesh.h"
#include "EventManager.h"
#include "PrimitiveTopology.h"
#include "CullingMode.h"
#include "VisualisationState.h"

CUDASceneGraph::CUDASceneGraph()
	: m_Visualisation{ VisualisationState::PBR }
	, m_SampleState{ SampleState::Point }
	, m_CullingMode{ CullingMode::BackFace }
	, m_pCudaMeshes{}
{}

CUDASceneGraph::~CUDASceneGraph()
{
	RemoveAllCUDAMeshes();
}

void CUDASceneGraph::LoadSceneGraph(SceneGraph* pSceneGraph, const CUDATextureManager& tm)
{
	if (!pSceneGraph)
	{
		std::cout << "!CUDASceneGraph::LoadSceneGraph > Invalid scenegraph!\n";
		return;
	}

	RemoveAllCUDAMeshes();
	const std::vector<Mesh*>& pMeshes = pSceneGraph->GetMeshes();
	for (Mesh* pMesh : pMeshes)
	{
		CUDAMesh* pCudaMesh = AddCUDAMesh(pMesh);
		const int* pTexIds = pMesh->GetTextureIds();
		UpdateCUDAMeshTextures(pCudaMesh, pTexIds, tm);
	}
}

void CUDASceneGraph::Update(float elapsedSec)
{
	constexpr float rotateSpeed = 1.f;
	const FMatrix4 rotationMatrix = (FMatrix4)MakeRotationY(rotateSpeed * elapsedSec);
	for (CUDAMesh* pCudaMesh : m_pCudaMeshes)
	{
		FMatrix4& world = pCudaMesh->GetWorld();
		world *= rotationMatrix;
	}

	//Handle Input Events
	if (EventManager::IsKeyPressed(SDLK_r))
	{
		ToggleVisualisation();
	}
	if (EventManager::IsKeyPressed(SDLK_f))
	{
		ToggleSampleState();
	}
	if (EventManager::IsKeyPressed(SDLK_c))
	{
		ToggleCullingMode();
	}
}

unsigned int CUDASceneGraph::GetTotalNumTriangles() const
{
	unsigned int numTriangles{};
	for (CUDAMesh* pCudaMesh : m_pCudaMeshes)
	{
		numTriangles += pCudaMesh->GetTotalNumTriangles();
	}
	return numTriangles;
}

void CUDASceneGraph::ToggleVisualisation()
{
	m_Visualisation = VisualisationState((int)m_Visualisation + 1);
	if ((int)m_Visualisation > 2)
		m_Visualisation = VisualisationState::PBR;

	switch (m_Visualisation)
	{
	case VisualisationState::PBR:
		std::cout << "\nVisualising PBR Colour\n";
		break;
	case VisualisationState::Depth:
		std::cout << "\nVisualising Depth Colour\n";
		break;
	case VisualisationState::Normal:
		std::cout << "\nVisualising Normal Colour\n";
		break;
	}
}

void CUDASceneGraph::ToggleSampleState()
{
	m_SampleState = SampleState((int)m_SampleState + 1);
	if ((int)m_SampleState > 1)
		m_SampleState = SampleState::Point;

	switch (m_SampleState)
	{
	case SampleState::Point:
		std::cout << "\nDefaultTechnique (Point)\n";
		break;
	case SampleState::Linear:
		std::cout << "\nLinearTechnique\n";
		break;
	}
}

void CUDASceneGraph::ToggleCullingMode()
{
	m_CullingMode = CullingMode((int)m_CullingMode + 1);
	if ((int)m_CullingMode > 2)
		m_CullingMode = CullingMode::BackFace;

	switch (m_CullingMode)
	{
	case CullingMode::BackFace:
		std::cout << "\nBackFace Culling\n";
		break;
	case CullingMode::FrontFace:
		std::cout << "\nFrontFace Culling\n";
		break;
	case CullingMode::NoCulling:
		std::cout << "\nNo Culling\n";
		break;
	}
}

CPU_CALLABLE
CUDAMesh* CUDASceneGraph::AddCUDAMesh(const Mesh* pMesh)
{
	CUDAMesh* pCudaMesh = new CUDAMesh{ pMesh };
	m_pCudaMeshes.push_back(pCudaMesh);
	return pCudaMesh;
}

CPU_CALLABLE
void CUDASceneGraph::RemoveAllCUDAMeshes()
{
	for (CUDAMesh* pCudaMesh : m_pCudaMeshes)
	{
		delete pCudaMesh;
	}
	m_pCudaMeshes.clear();
}

CPU_CALLABLE
void CUDASceneGraph::RemoveCUDAMesh(CUDAMesh* pCudaMesh)
{
	auto it = std::find(m_pCudaMeshes.cbegin(), m_pCudaMeshes.cend(), pCudaMesh);
	if (it != m_pCudaMeshes.cend())
	{
		delete pCudaMesh;
		m_pCudaMeshes.erase(it);
	}
}

CPU_CALLABLE
CUDATexturesCompact CUDASceneGraph::GetCUDAMeshTextures(const int* texIds, const CUDATextureManager& tm)
{
	//Preload textures and fetch instead of creating a TexturesCompact object every frame in Render()
	//Instead of textures being fetched, alter them in CUDAMesh object
	//The actual TexuresCompact object gets copied to the kernel anyway (POD GPU data and ptrs)
	const CUDATexture* pDiff = tm.GetCUDATexture(texIds[Mesh::TextureID::Diffuse]);
	const CUDATexture* pNorm = tm.GetCUDATexture(texIds[Mesh::TextureID::Normal]);
	const CUDATexture* pSpec = tm.GetCUDATexture(texIds[Mesh::TextureID::Specular]);
	const CUDATexture* pGloss = tm.GetCUDATexture(texIds[Mesh::TextureID::Glossiness]);
	CUDATexturesCompact textures{};
	if (pDiff)
	{
		textures.Diff = CUDATextureCompact::CompactCUDATexture(*pDiff);
		if (pNorm)
		{
			textures.Norm = CUDATextureCompact::CompactCUDATexture(*pNorm);
			if (pSpec)
				textures.Spec = CUDATextureCompact::CompactCUDATexture(*pSpec);
			if (pGloss)
				textures.Gloss = CUDATextureCompact::CompactCUDATexture(*pGloss);
		}
	}
	return textures;
}

CPU_CALLABLE
void CUDASceneGraph::UpdateCUDAMeshTextures(CUDAMesh* pCudaMesh, const int texIds[4], const CUDATextureManager& tm)
{
	const CUDATexturesCompact textures = GetCUDAMeshTextures(texIds, tm);
	pCudaMesh->SetTextures(textures);
}