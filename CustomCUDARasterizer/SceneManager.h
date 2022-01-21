#pragma once
#include <vector>
#include "Texture.h"

class SceneGraph;
class SceneManager final
{
public:
	SceneManager();
	~SceneManager();

	void Update(float elapsedSec);

	SceneGraph* AddSceneGraph(SceneGraph* pSceneGraph);

	bool IsDepthColour() const;
	SceneGraph* GetSceneGraph() const;
	SampleState GetSampleState() const;

	void ChangeSceneGraph();
	void ChangeSceneGraph(int idx);
	void ToggleDepthColour();
	void ToggleSampleState();

private:
	bool m_IsDepthColour;
	int m_Index;
	std::vector<SceneGraph*> m_pSceneGraphs;
	SampleState m_SampleState;
};