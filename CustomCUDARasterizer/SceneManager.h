#pragma once
#include <vector>
#include "Texture.h"

class SceneGraph;
enum class SampleState;
enum class CullingMode;
enum class VisualisationState;
class SceneManager final
{
public:
	SceneManager();
	~SceneManager();

	void Update(float elapsedSec);

	void AddSceneGraph(SceneGraph* pSceneGraph);

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
	std::vector<SceneGraph*> m_pSceneGraphs;
	SampleState m_SampleState;
	CullingMode m_CullingMode;
};