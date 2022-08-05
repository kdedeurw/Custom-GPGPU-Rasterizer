#pragma once
#include <vector>

class Mesh;
class Light;
class SceneGraph
{
public:
	SceneGraph() = default;
	virtual ~SceneGraph();

	void AddMesh(Mesh* pMesh);
	Light* AddLight(Light* pLight);

	const std::vector<Mesh*>& GetMeshes() const { return m_pMeshes; }
	const std::vector<Light*>& GetLights() const { return m_pLights; }

	void Update(float elapsedSec);

protected:
	std::vector<Mesh*> m_pMeshes;
	std::vector<Light*> m_pLights;
};