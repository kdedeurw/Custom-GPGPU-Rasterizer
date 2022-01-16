#pragma once
#include <vector>

class Mesh;
class Light;
class SceneGraph
{
public:
	SceneGraph();
	explicit SceneGraph(const std::vector<Mesh*>& pMeshes);
	~SceneGraph();

	Mesh* AddMesh(Mesh* pMesh);
	Light* AddLight(Light* pLight);

	const std::vector<Mesh*>& GetObjects() const;
	const std::vector<Light*>& GetLights() const;

	void Update(float elapsedSec);

private:
	std::vector<Mesh*> m_pMeshes;
	std::vector<Light*> m_pLights;
};