#include "SceneManager.h"
#include <iostream>
#include "SceneGraph.h"
#include "Mesh.h"

SceneManager* SceneManager::m_pSceneManager{ nullptr };

SceneManager::SceneManager()
	: m_pSceneGraphs{}
	, m_Index{4}
	, m_SampleState{ SampleState::Point }
{}

SceneManager::~SceneManager()
{
	for (SceneGraph* pSceneGraph : m_pSceneGraphs)
	{
		delete pSceneGraph;
	}
	m_pSceneGraphs.clear();

	m_pSceneManager = nullptr;
}

SceneGraph* SceneManager::AddSceneGraph(SceneGraph* pSceneGraph)
{
	m_pSceneGraphs.push_back(pSceneGraph);
	return m_pSceneGraphs.back();
}

SceneGraph* SceneManager::GetSceneGraph() const
{
	if (!m_pSceneGraphs.size())
	{
		std::cout << "\n!Warning; no SceneGraph(s) detected in the SceneManager!\n";
		return nullptr;
	}
	return m_pSceneGraphs.at(m_Index);
}

SampleState SceneManager::GetSampleState() const
{
	return m_SampleState;
}

void SceneManager::ChangeSceneGraph()
{
	++m_Index;
	if (m_Index >= m_pSceneGraphs.size())
		m_Index = 0;
}

void SceneManager::Update(float elapsedSec)
{
	for (SceneGraph* pSceneGraph : m_pSceneGraphs)
	{
		pSceneGraph->Update(elapsedSec);
	}
}

void SceneManager::ToggleDepthColour()
{
	m_IsDepthColour = !m_IsDepthColour;
}

void SceneManager::ToggleSampleState()
{
	m_SampleState = SampleState((int)m_SampleState + 1);
	if ((int)m_SampleState > 1)
		m_SampleState = SampleState::Point;

	if (m_SampleState == SampleState::Point)
		std::cout << "\nDefaultTechnique (Point)\n";
	else
		std::cout << "\nLinearTechnique\n";
}

bool SceneManager::IsDepthColour() const
{
	return m_IsDepthColour;
}