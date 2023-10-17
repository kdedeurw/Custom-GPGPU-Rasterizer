#pragma once
#include <vector>
#include "CUDAStructs.h"

class SceneGraph;
enum class SampleState;
enum class CullingMode;
enum class VisualisationState;
class CUDAMesh;
class CUDATextureManager;

class CUDASceneGraph final
{
public:
	CUDASceneGraph();
	~CUDASceneGraph();

	//Preload and store scene in persistent memory
	//This will eliminate overhead by loading mesh data and accessing global memory
	void LoadSceneGraph(SceneGraph* pSceneGraph, const CUDATextureManager& tm);

	void Update(float elapsedSec);

	unsigned int GetTotalNumTriangles() const;

	VisualisationState GetVisualisationState() const { return m_Visualisation; };
	SampleState GetSampleState() const { return m_SampleState; };
	CullingMode GetCullingMode() const { return m_CullingMode; };
	const std::vector<CUDAMesh*>& GetCUDAMeshes() const { return m_pCudaMeshes; };

	void ToggleVisualisation();
	void ToggleSampleState();
	void ToggleCullingMode();

private:
	VisualisationState m_Visualisation;
	SampleState m_SampleState;
	CullingMode m_CullingMode;
	std::vector<CUDAMesh*> m_pCudaMeshes;

	//function that allocates and copies host mesh buffers to device
	CUDAMesh* AddCUDAMesh(const Mesh* pMesh);
	//function that fetches a mesh's textures
	CUDATexturesCompact GetCUDAMeshTextures(const int* texIds, const CUDATextureManager& tm);
	//function that pre-stores a mesh's textures
	void UpdateCUDAMeshTextures(CUDAMesh* pCudaMesh, const int texIds[4], const CUDATextureManager& tm);
	//function that frees all device mesh buffers
	void RemoveAllCUDAMeshes();
	//function that frees a device mesh buffers
	void RemoveCUDAMesh(CUDAMesh* pCudaMesh);

};