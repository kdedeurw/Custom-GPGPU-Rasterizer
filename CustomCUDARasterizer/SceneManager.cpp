#include "PCH.h"
#include "SceneManager.h"
#include <iostream>
#include "SceneGraph.h"
#include "Mesh.h"
#include "EventManager.h"

SceneManager::SceneManager()
	: m_IsDepthColour{}
	, m_pSceneGraphs{}
	, m_Index{}
	, m_SampleState{ SampleState::Point }
{}

SceneManager::~SceneManager()
{
	for (SceneGraph* pSceneGraph : m_pSceneGraphs)
	{
		delete pSceneGraph;
	}
	m_pSceneGraphs.clear();
}

SceneGraph* SceneManager::AddSceneGraph(SceneGraph* pSceneGraph)
{
	m_pSceneGraphs.push_back(pSceneGraph);
	return m_pSceneGraphs.back();
}

SceneGraph* SceneManager::GetSceneGraph() const
{
	const size_t size = m_pSceneGraphs.size();
	if (size == 0)
	{
		std::cout << "\n!Warning; no SceneGraph(s) detected in the SceneManager!\n";
		return nullptr;
	}
	else if (m_Index >= size)
		return m_pSceneGraphs.at(size - 1);
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

void SceneManager::ChangeSceneGraph(int idx)
{
	if (idx > 0 && idx < m_pSceneGraphs.size())
		m_Index = idx;
}

void SceneManager::Update(float elapsedSec)
{
	for (SceneGraph* pSceneGraph : m_pSceneGraphs)
	{
		pSceneGraph->Update(elapsedSec);
	}

	//Handle Input Events
	if (EventManager::IsKeyPressed(SDLK_TAB) || EventManager::IsKeyPressed(SDLK_MINUS))
	{
		ChangeSceneGraph();
	}
	if (EventManager::IsKeyPressed(SDLK_r))
	{
		ToggleDepthColour();
	}
	if (EventManager::IsKeyPressed(SDLK_f))
	{
		ToggleSampleState();
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