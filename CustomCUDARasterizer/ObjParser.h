#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "Math.h"
#include "Vertex.h"

class ObjParser
{
	static const char* OBJ_EXTENSION;
	struct Indexed
	{
		unsigned int v{}, vt{}, vn{};
		unsigned int idx{};
	};
public:
	ObjParser();
	virtual ~ObjParser();

	bool OpenFile(const std::string& filePath);
	void CloseFile();

	void ReadFromObjFile(std::vector<IVertex>& vertexBuffer, std::vector<unsigned int>& indexBuffer);

	void SetInvertYAxis(bool value);

private:
	bool m_IsYAxisInverted;
	std::ifstream m_ReadFile;

	std::vector<FPoint3> m_Positions;
	std::vector<FVector2> m_UVs;
	std::vector<FVector3> m_Normals;
	std::vector<unsigned int> m_PositionIndices;
	std::vector<unsigned int> m_UVIndices;
	std::vector<unsigned int> m_NormalIndices;

	void StorePosition(std::stringstream& position);
	void StoreFace(std::stringstream& face);
	void StoreNormal(std::stringstream& normal);
	void StoreUV(std::stringstream& uv);
	void GetFirstSecondThird(std::stringstream& fst, std::string& first, std::string& second, std::string& third);

	void AssignVertices(std::vector<IVertex>& vertexBuffer, std::vector<unsigned int>& indexBuffer);
	void ClearData();
};