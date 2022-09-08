#pragma once
#include <vector>
#include "Texture.h"

class SceneGraph;
enum class SampleState;
enum class CullingMode;
class SceneManager final
{
public:
	SceneManager();
	~SceneManager();

	void Update(float elapsedSec);

	void AddSceneGraph(SceneGraph* pSceneGraph);

	bool IsDepthColour() const { return m_IsDepthColour; };
	SceneGraph* GetSceneGraph() const;
	SampleState GetSampleState() const { return m_SampleState; };
	CullingMode GetCullingMode() const { return m_CullingMode; };

	void ChangeSceneGraph();
	void ChangeSceneGraph(int idx);
	void ToggleDepthColour();
	void ToggleSampleState();
	void ToggleCullingMode();

private:
	bool m_IsDepthColour;
	int m_Index;
	std::vector<SceneGraph*> m_pSceneGraphs;
	SampleState m_SampleState;
	CullingMode m_CullingMode;
};