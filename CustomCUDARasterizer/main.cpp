#include "PCH.h"

//External includes
#include "vld.h"
#include "SDL.h"
#undef main

//Project includes
#include "EventManager.h"
#include "ObjParser.h"
#include "DirectionalLight.h"
#include "CUDATextureManager.h"

//Project CUDA includes
#include "Application.h"

//#include <curand_kernel.h>
//#include <curand.h>

int AddTexture(CUDATextureManager& tm, const std::string& texPath)
{
	if (texPath.empty())
		return -1;

	CUDATexture* pCUDATexture = new CUDATexture{};
	pCUDATexture->Create(texPath.c_str());
	if (!pCUDATexture->IsAllocated())
	{
		std::cout << "!Error: AddTexture > Texture is invalid and not allocated! (Wrong path?)\n";
		std::cout << "Path: \"" << texPath << "\"\n";
		return -1;
	}
	return tm.AddCUDATexture(pCUDATexture);
}

void CreateScenes(SceneManager& sm, CUDATextureManager& tm)
{
	std::vector<SceneGraph*> pSceneGraphs{};
	ObjParser parser{};
	std::vector<IVertex> vertexBuffer{};
	std::vector<unsigned int> indexBuffer{};
	short vertexType{ -1 }, vertexStride{ sizeof(IVertex) };

	//{
	//	// SceneGraph 1
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // White small Triangle
	//		std::vector<IVertex> vertices = {
	//			{ FPoint3{ 0.f, 2.f, 0.f }, RGBColor{1.f, 1.f, 1.f} },
	//			{ FPoint3{ -1.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} },
	//			{ FPoint3{ 1.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} } };
	//		std::vector<unsigned int> indices = { 0, 1, 2 };
	//		Mesh* pTriangle = new Mesh{ vertices, vertexStride, vertexType, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pTriangle);
	//	}
	//	{
	//		// Mesh 2 // Coloured larger Triangle
	//		std::vector<IVertex> vertices = {
	//			{ FPoint3{ 0.f, 4.f, -2.f }, RGBColor{1.f, 0.f, 0.f} },
	//			{ FPoint3{ -3.f, -2.f, -2.f }, RGBColor{0.f, 1.f, 0.f} },
	//			{ FPoint3{ 3.f, -2.f, -2.f }, RGBColor{0.f, 0.f, 1.f} } };
	//		std::vector<unsigned int> indices = { 0, 1, 2 };
	//		Mesh* pTriangle = new Mesh{ vertices, vertexStride, vertexType, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pTriangle);
	//	}
	//	//pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	//{
	//	// SceneGraph 2
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	short vertexType{};
	//	vertexType |= (int)VertexType::Uv;
	//	CUDATexture* pUVGridTexture = new CUDATexture{ "Resources/uv_grid_2.png" };
	//	if (!pUVGridTexture->IsAllocated())
	//	{
	//		throw std::runtime_error{ "CreateScenes > pUVGridTexture was unable to allocate" };
	//	}
	//	const int uvGridTextureId = AddTexture(tm, "Resources/uv_grid_2.png");
	//	{
	//		numVertices = 9;
	//		numIndices = 24;
	//		// Mesh 1 // TriangleList Quad
	//		std::vector<IVertex> vertices{
	//		{ FPoint3{-3, 3, -2}, FVector2{0, 0} },
	//		{ FPoint3{ 0, 3, -2 }, FVector2{ 0.5f, 0 } },
	//		{ FPoint3{ 3, 3, -2 }, FVector2{ 1, 0 } },
	//		{ FPoint3{ -3, 0, -2 }, FVector2{ 0, 0.5f } },
	//		{ FPoint3{ 0, 0, -2 }, FVector2{ 0.5f, 0.5f } },
	//		{ FPoint3{ 3, 0, -2 }, FVector2{ 1, 0.5f } },
	//		{ FPoint3{ -3, -3, -2 }, FVector2{ 0, 1 } },
	//		{ FPoint3{ 0, -3, -2 }, FVector2{ 0.5f, 1 } },
	//		{ FPoint3{ 3, -3, -2 }, FVector2{ 1, 1 } }, };
	//		std::vector<unsigned int> indices{
	//		0,3,1, 3,4,1, 1,4,2, 4,5,2, 3,6,4, 6,7,4, 4,7,5, 7,8,5, };
	//		Mesh* pTriangleListQuad = new Mesh{ vertices, vertexStride, vertexType, indices, PrimitiveTopology::TriangleList };
	//		pTriangleListQuad->SetTextureId(uvGridTextureId, Mesh::TextureID::Diffuse);
	//		pSceneGraph->AddMesh(pTriangleListQuad);
	//	}
	//	//{
	//	//	// Mesh 2 // TriangleStrip Quad
	//	//	std::vector<IVertex> vertices = {
	//	//		{ FPoint3{-3, 3, -2}, FVector2{0, 0} }, IVertex{ FPoint3{0, 3, -2}, FVector2{0.5f, 0} }, IVertex{ FPoint3{3, 3, -2}, FVector2{1, 0} },
	//	//		{ FPoint3{-3, 0, -2}, FVector2{0, 0.5f} }, IVertex{ FPoint3{0, 0, -2}, FVector2{0.5f, 0.5f} }, IVertex{ FPoint3{3, 0, -2}, FVector2{1, 0.5f} },
	//	//		{ FPoint3{-3, -3, -2}, FVector2{0, 1} }, IVertex{ FPoint3{0, -3, -2}, FVector2{0.5f, 1} }, IVertex{ FPoint3{3, -3, -2}, FVector2{1, 1} } };
	//	//	std::vector<unsigned int> indices = { 0, 3, 1, 4, 2, 5, 5, 3, 3, 6, 4, 7, 5, 8 }; // strip
	//	//	Mesh* pTriangleStripQuad = new Mesh{ vertices, vertexStride, vertexType, indices, PrimitiveTopology::TriangleStrip };
	//	//	pTriangleStripQuad->SetTextureId(uvGridTextureId, Mesh::TextureID::Diffuse);
	//	//	pSceneGraph->AddMesh(pTriangleStripQuad);
	//	//}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}
	
	//{
	//	// SceneGraph 3 // TukTuk
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // TukTuk
	//		short vertexType{};
	//		parser.OpenFile("Resources/tuktuk.obj");
	//		parser.SetInvertYAxis(true);
	//		parser.ParseObjFile(vertexBuffer, indexBuffer, vertexType);
	//		Mesh* pTukTukMesh = new Mesh{ vertexBuffer, vertexStride, vertexType, indexBuffer, PrimitiveTopology::TriangleList };
	//		const int diffTexId = AddTexture(tm, "Resources/tuktuk.png");
	//		pTukTukMesh->SetTextureId(diffTexId, Mesh::TextureID::Diffuse);
	//		pSceneGraph->AddMesh(pTukTukMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//
	//}

	{
		// SceneGraph 4 // Bunny
		SceneGraph* pSceneGraph = new SceneGraph{};
		{
			// Mesh 1 // Bunny 
			parser.OpenFile("Resources/lowpoly_bunny.obj");
			parser.ParseObjFile(vertexBuffer, indexBuffer, vertexType);
			Mesh* pBunnyMesh = new Mesh{ vertexBuffer, vertexStride, vertexType, indexBuffer, PrimitiveTopology::TriangleList };
			pSceneGraph->AddMesh(pBunnyMesh);
		}
		pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
		pSceneGraphs.push_back(pSceneGraph);
	}
	
	//{
	//	// SceneGraph 5 // Vehicle
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1 // Vehicle
	//		short vertexType{};
	//		parser.OpenFile("Resources/vehicle.obj");
	//		parser.SetInvertYAxis(true);
	//		parser.ParseObjFile(vertexBuffer, indexBuffer, vertexType);
	//		Mesh* pVehicleMesh = new Mesh{ vertexBuffer, vertexStride, vertexType, indexBuffer, PrimitiveTopology::TriangleList };
	//		pVehicleMesh->SetTextureId(AddTexture(tm, "Resources/vehicle_diffuse.png"), Mesh::TextureID::Diffuse);
	//		pVehicleMesh->SetTextureId(AddTexture(tm, "Resources/vehicle_normal.png"), Mesh::TextureID::Normal);
	//		pVehicleMesh->SetTextureId(AddTexture(tm, "Resources/vehicle_specular.png"), Mesh::TextureID::Specular);
	//		pVehicleMesh->SetTextureId(AddTexture(tm, "Resources/vehicle_gloss.png"), Mesh::TextureID::Glossiness);
	//		pSceneGraph->AddMesh(pVehicleMesh);
	//	}
	//	pSceneGraph->AddLight(new DirectionalLight{ RGBColor{1.f, 1.f, 1.f}, 2.f, FVector3{ 0.577f, -0.577f, -0.577f } });
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	//{
	//	// SceneGraph 6 // Cube
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1
	//		std::vector<IVertex> vertices = {
	//		{ FPoint3{ 1.f, -1.f, -1.f }, RGBColor{1.f, 0.f, 0.f} },
	//		{ FPoint3{ 1.f, -1.f, 1.f }, RGBColor{0.f, 1.f, 0.f} },
	//		{ FPoint3{ -1.f, -1.f, 1.f }, RGBColor{0.f, 0.f, 1.f} },
	//		{ FPoint3{ -1.f, -1.f, -1.f }, RGBColor{1.f, 0.f, 0.f} },
	//		{ FPoint3{ 1.f, 1.f, -1.f }, RGBColor{0.f, 1.f, 0.f} },
	//		{ FPoint3{ 1.f, 1.f, 1.f }, RGBColor{0.f, 0.f, 1.f} },
	//		{ FPoint3{ -1.f, 1.f, 1.f }, RGBColor{1.f, 0.f, 0.f} },
	//		{ FPoint3{ -1.f, 1.f, -1.f }, RGBColor{0.f, 1.f, 0.f} } };
	//		std::vector<unsigned int> indices = {
	//			1, 2, 3, 7, 6, 5, 4, 5, 1, 5, 6, 2, 2, 6, 7, 0, 3, 7, 0, 1, 3, 4, 7, 5, 0, 4, 1, 1, 5, 2, 3, 2, 7, 4, 0, 7, //BackFace
	//			7, 0, 4, 7, 2, 3, 2, 5, 1, 1, 4, 0, 5, 7, 4, 3, 1, 0, 7, 3, 0, 7, 6, 2, 2, 6, 5, 1, 5, 4, 5, 6, 7, 3, 2, 1 }; //FrontFace
	//		Mesh* pCube = new Mesh{ vertices, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pCube);
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	//{
	//	// SceneGraph 7 // VertexCount
	//	SceneGraph* pSceneGraph = new SceneGraph{};
	//	{
	//		// Mesh 1
	//		const unsigned int numVertices = 3 * 1000000;
	//		std::vector<IVertex> vertices{};
	//		std::vector<unsigned int> indices{};
	//		vertices.reserve(numVertices);
	//		indices.reserve(numVertices * 3);
	//		for (int i{}; i < numVertices; i += 3)
	//		{
	//			vertices.push_back({ FPoint3{ 0.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} });
	//			vertices.push_back({ FPoint3{ 0.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} });
	//			vertices.push_back({ FPoint3{ 0.f, 0.f, 0.f }, RGBColor{1.f, 1.f, 1.f} });
	//			indices.push_back(i);
	//			indices.push_back(i + 1);
	//			indices.push_back(i + 2);
	//		}
	//		Mesh* pVertices = new Mesh{ vertices, indices, PrimitiveTopology::TriangleList };
	//		pSceneGraph->AddMesh(pVertices);
	//	}
	//	pSceneGraphs.push_back(pSceneGraph);
	//}

	for (SceneGraph* pSceneGraph : pSceneGraphs)
	{
		sm.AddSceneGraph(pSceneGraph);
	}
}

int main(int argc, char* args[])
{
	//Unreferenced parameters
	(void)argc;
	(void)args;

	//Camera Setup
	//const FPoint3 camPos = { 0.f, 5.f, 65.f };
	const FPoint3 camPos = { 0.f, 1.f, 5.f };
	const float fov = 45.f;
	Camera camera{ camPos, fov };

	Application app{camera};
	if (!app.Init(0, Resolution::ResolutionStandard::SD))
	{
		std::cout << "Application unable to initialize!\n";
		return -1;
	}

	{
		SceneManager sm{};
		CUDATextureManager& tm = app.GetTextureManager();
		CreateScenes(sm, tm);
		app.LoadSceneGraph(sm.GetSceneGraph());
	}

	app.Run();

	return 0;
}