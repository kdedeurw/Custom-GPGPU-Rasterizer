#include "PCH.h"
#include "SceneManager.h"
#include <iostream>
#include "SceneGraph.h"
#include "Mesh.h"
#include "EventManager.h"
#include "PrimitiveTopology.h"
#include "CullingMode.h"
#include "VisualisationState.h"

SceneManager::SceneManager()
	: m_Index{ 0 }
	, m_Visualisation{ VisualisationState::PBR }
	, m_SampleState{ SampleState::Point }
	, m_CullingMode{ CullingMode::BackFace }
	, m_pSceneGraphs{}
{}

SceneManager::~SceneManager()
{
	for (SceneGraph* pSceneGraph : m_pSceneGraphs)
	{
		delete pSceneGraph;
	}
	m_pSceneGraphs.clear();
}

void SceneManager::AddSceneGraph(SceneGraph* pSceneGraph)
{
	m_pSceneGraphs.push_back(pSceneGraph);
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
	if (EventManager::IsKeyPressed(SDLK_TAB))
	{
		ChangeSceneGraph();
	}
	else if (EventManager::IsKeyPressed(SDLK_MINUS))
	{
		ChangeSceneGraph(m_Index - 1);
	}
	if (EventManager::IsKeyPressed(SDLK_r))
	{
		ToggleVisualisation();
	}
	if (EventManager::IsKeyPressed(SDLK_f))
	{
		ToggleSampleState();
	}
	if (EventManager::IsKeyPressed(SDLK_c))
	{
		ToggleCullingMode();
	}
}

void SceneManager::ToggleVisualisation()
{
	m_Visualisation = VisualisationState((int)m_Visualisation + 1);
	if ((int)m_Visualisation > 2)
		m_Visualisation = VisualisationState::PBR;

	switch (m_Visualisation)
	{
	case VisualisationState::PBR:
		std::cout << "\nVisualising PBR Colour\n";
		break;
	case VisualisationState::Depth:
		std::cout << "\nVisualising Depth Colour\n";
		break;
	case VisualisationState::Normal:
		std::cout << "\nVisualising Normal Colour\n";
		break;
	}
}

void SceneManager::ToggleSampleState()
{
	m_SampleState = SampleState((int)m_SampleState + 1);
	if ((int)m_SampleState > 1)
		m_SampleState = SampleState::Point;

	switch (m_SampleState)
	{
	case SampleState::Point:
		std::cout << "\nDefaultTechnique (Point)\n";
		break;
	case SampleState::Linear:
		std::cout << "\nLinearTechnique\n";
		break;
	}
}

void SceneManager::ToggleCullingMode()
{
	m_CullingMode = CullingMode((int)m_CullingMode + 1);
	if ((int)m_CullingMode > 2)
		m_CullingMode = CullingMode::BackFace;

	switch (m_CullingMode)
	{
	case CullingMode::BackFace:
		std::cout << "\nBackFace Culling\n";
		break;
	case CullingMode::FrontFace:
		std::cout << "\nFrontFace Culling\n";
		break;
	case CullingMode::NoCulling:
		std::cout << "\nNo Culling\n";
		break;
	}
}