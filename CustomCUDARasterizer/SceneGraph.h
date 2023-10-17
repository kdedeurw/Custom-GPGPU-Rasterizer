#pragma once
#include <vector>

//TODO: template of SceneGraph?
//TODO: base class of SceneGraph?
//TODO: base class of Mesh?
//TODO: template of Mesh?

class Mesh;
class Light;
//template<typename T>
class SceneGraph
{
public:
	SceneGraph() = default;
	virtual ~SceneGraph();

	void AddMesh(Mesh* pMesh);
	void AddLight(Light* pLight);

	void Update(float elapsedSec);

	const std::vector<Mesh*>& GetMeshes() const { return m_pMeshes; }
	const std::vector<Light*>& GetLights() const { return m_pLights; }

	unsigned int GetTotalNumTriangles() const;

protected:
	std::vector<Mesh*> m_pMeshes{};
	std::vector<Light*> m_pLights{};
};