#include "PCH.h"
#include "SceneGraph.h"
#include "Mesh.h"
#include "Light.h"

SceneGraph::~SceneGraph()
{
	for (Mesh* pCMesh : m_pMeshes)
	{
		delete pCMesh;
	}
	m_pMeshes.clear();
	for (Light* pLight : m_pLights)
	{
		delete pLight;
	}
	m_pLights.clear();
}

void SceneGraph::AddMesh(Mesh* pMesh)
{
	m_pMeshes.push_back(pMesh);
}

void SceneGraph::AddLight(Light* pLight)
{
	m_pLights.push_back(pLight);
}

void SceneGraph::Update(float elapsedSec)
{
	//TODO: refactor this code
	constexpr float rotateSpeed = 1.f;
	const FMatrix4 rotationMatrix = (FMatrix4)MakeRotationY(rotateSpeed * elapsedSec);
	for (Mesh* pMesh : m_pMeshes)
	{
		FMatrix4& world = pMesh->GetWorld();
		world *= rotationMatrix;
	}
}