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

Light* SceneGraph::AddLight(Light* pLight)
{
	m_pLights.push_back(pLight);
	return pLight;
}

void SceneGraph::Update(float elapsedSec)
{
	std::for_each(m_pMeshes.begin(), m_pMeshes.end(), [elapsedSec](Mesh* pMesh) { pMesh->Update(elapsedSec); });
}