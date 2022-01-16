#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "ERGBColor.h"
#include "EMath.h"
#include "EMathUtilities.h"
#include "Vertex.h"

class ObjParser
{
public:
	ObjParser();
	ObjParser(const std::string& filePath);
	~ObjParser();

	bool OpenFile(const std::string& filePath);
	void CloseFile();

	void ReadFromObjFile();
	void ReadFromObjFile(const std::string& filePath);

	std::vector<IVertex> const * GetVertexBuffer() const;
	const std::vector<int> GetIndexBuffer() const;

	void SetInvertYAxis(bool value);

private:
	std::ifstream m_ReadFile;

	void StorePosition(std::stringstream& position);
	void StoreFace(std::stringstream& face);
	void StoreNormal(std::stringstream& normal);
	void StoreUV(std::stringstream& uv);
	void GetFirstSecondThird(std::stringstream& fst, std::string& first, std::string& second, std::string& third);

	void AssignVertices();

	struct Indexed
	{
		int v{}, vt{}, vn{};
		int idx{};
	};

	bool m_IsYAxisInverted;

	std::vector<IVertex>* m_pVertices;
	std::vector<int> m_IndexBuffer;

	std::vector<FPoint3> m_Positions;
	std::vector<FVector2> m_UVs;
	std::vector<FVector3> m_Normals;
	std::vector<int> m_PositionIndices;
	std::vector<int> m_UVIndices;
	std::vector<int> m_NormalIndices;
};