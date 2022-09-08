#pragma once
#include <vector>

struct Mesh;
class Light;
class SceneGraph
{
public:
	SceneGraph() = default;
	virtual ~SceneGraph();

	void AddMesh(Mesh* pMesh);
	void AddLight(Light* pLight);

	const std::vector<Mesh*>& GetMeshes() const { return m_pMeshes; }
	const std::vector<Light*>& GetLights() const { return m_pLights; }

	void Update(float elapsedSec);

protected:
	std::vector<Mesh*> m_pMeshes;
	std::vector<Light*> m_pLights;
};