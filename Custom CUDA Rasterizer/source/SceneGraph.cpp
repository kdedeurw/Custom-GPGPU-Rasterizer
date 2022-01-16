#include "SceneGraph.h"
#include "Mesh.h"

SceneGraph::SceneGraph()
	: SceneGraph{ std::vector<Mesh*>{} }
{}

SceneGraph::SceneGraph(const std::vector<Mesh*>& pMeshes)
	: m_pMeshes{ pMeshes }
{}

SceneGraph::~SceneGraph()
{
	for (Mesh* pMesh : m_pMeshes)
	{
		delete pMesh;
	}
	m_pMeshes.clear();
}

Mesh* SceneGraph::AddMesh(Mesh* pMesh)
{
	m_pMeshes.push_back(pMesh);
	return pMesh;
}

Light* SceneGraph::AddLight(Light* pLight)
{
	m_pLights.push_back(pLight);
	return pLight;
}

const std::vector<Mesh*>& SceneGraph::GetObjects() const
{
	return m_pMeshes;
}

const std::vector<Light*>& SceneGraph::GetLights() const
{
	return m_pLights;
}

void SceneGraph::Update(float elapsedSec)
{
	std::for_each(m_pMeshes.begin(), m_pMeshes.end(), [elapsedSec](Mesh* pMesh) { pMesh->Update(elapsedSec); });
}