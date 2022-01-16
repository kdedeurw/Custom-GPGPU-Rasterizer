#pragma once
#include <vector>
#include "Texture.h"

class SceneGraph;

class SceneManager
{
public:
	static SceneManager* GetInstance()
	{
		if (!m_pSceneManager) m_pSceneManager = new SceneManager{};
		return m_pSceneManager;
	}
	~SceneManager();

	SceneGraph* AddSceneGraph(SceneGraph* pSceneGraph);

	bool IsDepthColour() const;
	SceneGraph* GetSceneGraph() const;
	SampleState GetSampleState() const;

	void ChangeSceneGraph();
	void Update(float elapsedSec);
	void ToggleDepthColour();
	void ToggleSampleState();

private:
	SceneManager();
	static SceneManager* m_pSceneManager;

	bool m_IsDepthColour{};
	size_t m_Index;
	std::vector<SceneGraph*> m_pSceneGraphs;
	SampleState m_SampleState;
};