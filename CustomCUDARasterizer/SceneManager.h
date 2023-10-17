#pragma once
#include <vector>

class SceneGraph;
enum class SampleState;
enum class CullingMode;
enum class VisualisationState;

class SceneManager final
{
public:
	SceneManager();
	~SceneManager();

	void AddSceneGraph(SceneGraph* pSceneGraph);

	void Update(float elapsedSec);

	VisualisationState GetVisualisationState() const { return m_Visualisation; };
	SceneGraph* GetSceneGraph() const;
	SampleState GetSampleState() const { return m_SampleState; };
	CullingMode GetCullingMode() const { return m_CullingMode; };

	void ChangeSceneGraph();
	void ChangeSceneGraph(int idx);
	void ToggleVisualisation();
	void ToggleSampleState();
	void ToggleCullingMode();

private:
	int m_Index;
	VisualisationState m_Visualisation;
	SampleState m_SampleState;
	CullingMode m_CullingMode;
	std::vector<SceneGraph*> m_pSceneGraphs;

};