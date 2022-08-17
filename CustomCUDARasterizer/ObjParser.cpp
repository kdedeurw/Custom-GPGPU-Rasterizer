#include "PCH.h"
#include "ObjParser.h"
#include "VertexType.h"
#include <iostream>
#include <sstream>

const char* ObjParser::OBJ_EXTENSION = ".obj";

ObjParser::ObjParser()
	: m_IsYAxisInverted{}
	, m_ReadFile{}
	, m_Positions{}
	, m_PositionIndices{}
	, m_UVIndices{}
	, m_NormalIndices{}
{}

ObjParser::~ObjParser()
{
	CloseFile();
}

void ObjParser::ParseData()
{
	//Clear all buffers
	ClearData();

	if (m_ReadFile.is_open())
	{
		std::cout << "\n!Parsing .obj file!\n";

		std::string line{};
		size_t lineCount{};
		while (std::getline(m_ReadFile, line))
		{
			++lineCount;
			if (line.empty() || line.front() == '#') continue; // skip empty and comment lines
			std::string prefix{};
			std::getline(std::stringstream{ line }, prefix, ' '); // prefix always in front, delimited by space(s)
			if (prefix == "v")
			{
				std::stringstream position{ line.substr(3) }; // create substring without 'v  ' (v and 2 spaces)
				StorePosition(position);
			}
			else if (prefix == "f") // every line starting with 'f'
			{
				std::stringstream face{ line.substr(2) }; // create substring without 'f ' (f and 1 space)
				StoreFace(face);
			}
			else if (prefix == "vt")
			{
				std::stringstream uv{ line.substr(3) }; // create substring without 'vt ' (vt and 1 space)
				StoreUV(uv);
			}
			else if (prefix == "vn")
			{
				std::stringstream normal{ line.substr(3) }; // create substring without 'vn ' (vn and 1 space)
				StoreNormal(normal);
			}
			else
			{
				std::cout << "\n!Unknown prefix found at line: " << lineCount << ", prefix: \' " << prefix << " \'!\n";
				std::cout << "Full line: \' " << line << " \'\n";
			}
		}

		// sort all possible faces, which idx's are now listed in vector<int>'s, and link them with the values of the normals and uvs we collected
		//AssignUVs(); // create std::vector<FVector3> filled with UV coords (no need to store these separately)
		//AssignNormals(); // create std::vector<FVector3> filled with normals (no need to store these separately)

		std::cout << "\n!Done parsing!\n";
	}
	else
	{
		std::cout << "\n!Unable to open file!\n";
	}
}

void ObjParser::ReadFromObjFile(std::vector<IVertex>& vertexBuffer, std::vector<unsigned int>& indexBuffer, short& vertexType)
{
	ParseData();

	std::cout << "\n!Creating vertices based on parsed info!\n";
	// create vertices, filled with positions, normals, UV coords (and colours?) all at once (store them all in vertex)
	AssignVertices(vertexBuffer, indexBuffer, vertexType);

	std::cout << "\n!All done!\n";
}

void ObjParser::StorePosition(std::stringstream& position)
{
	std::string x{}, y{}, z{};
	GetFirstSecondThird(position, x, y, z);
	m_Positions.push_back(FPoint3{ std::stof(x), std::stof(y), std::stof(z) });
}

void ObjParser::StoreFace(std::stringstream& face)
{
	// faces are laid out like this:
	// f v1[/ vt1][/ vn1] v2[/ vt2][/ vn2] v3[/ vt3][/ vn3]
	std::string first{}, second{}, third{}; // first being v, second being vt, third being vn
	GetFirstSecondThird(face, first, second, third);
	// first now contains v1/vt1/vn1
	// second now contains v2/vt2/vn2
	// third now contains v3/vt3/vn3

	const std::string* faces[3]{ &first, &second, &third };

	std::vector<unsigned int> faceIndexes{};
	faceIndexes.reserve(3); // atleast 3 indexes (max 9?)

	std::string index{};
	for (int i{}; i < 3; ++i)
	{
		size_t slashPos{};
		std::stringstream temp{ *faces[i] }; // all 3 strings
		do
		{
			std::getline(temp, index, '/'); // all indexes, delimited by a '/' (slash) ((!IF POSSIBLE!))
			faceIndexes.push_back(std::stoi(index) - 1); // !!!MINUS ONE SINCE OBJ STARTS AT 1 INSTEAD OF 0!!!
			slashPos = faces[i]->find('/', slashPos + 1);
			// using i as index, still using current faces[i] in this loop
			// if there's no delimiter, break off operation after having stored the first face element (there HAS to be atleast 1)
			if (slashPos == std::string::npos) break; // no slash found, meaning only 1 face, break loop and onto next face[++i]
			temp = std::stringstream{ faces[i]->substr(slashPos + 1) }; // create new substring from faces
		} while (true); // while (slashPos != std::string::npos) could also be used, but performance boost by adding if statement and break
	}

	//for (int i{}; i < 3; ++i)
	//{
	//	std::stringstream temp{ *faces[i] }; // all 3 strings
	//	for (int j{}; j < 3; ++j)
	//	{
	//		std::getline(temp, index, '/'); // all 3 indexes, delimited by a '/' (slash)
	//		faceIndexes.push_back(std::stoi(index) - 1); // !!!MINUS ONE SINCE OBJ STARTS AT 1 INSTEAD OF 0!!!
	//		if (faces[i]->find('/') == std::string::npos) break; // using i as index, NOT j, still using current faces[i] in this loop
	//		// if there's no delimiter, break off operation after having stored the first face element (there HAS to be atleast 1 in this loop)
	//	}
	//}

	const size_t size{ faceIndexes.size() };
	for (size_t idx{}; idx < size; ++idx)
	{
		size_t temp{ idx % (size / 3) };
		switch (temp)
		{
		case 0: // first face is an index in the indexbuffer
			m_PositionIndices.push_back(faceIndexes[idx]);
			break;
		case 1: // second face is a uv index
			m_UVIndices.push_back(faceIndexes[idx]);
			break;
		case 2: // third face is a normal index
			m_NormalIndices.push_back(faceIndexes[idx]);
			break;
		}
	}
}

void ObjParser::StoreNormal(std::stringstream& normal)
{
	std::string first{}, second{}, third{};
	GetFirstSecondThird(normal, first, second, third);
	m_Normals.push_back(GetNormalized(FVector3{ std::stof(first), std::stof(second), std::stof(third) }));
}

void ObjParser::StoreUV(std::stringstream& uv)
{
	std::string first{}, second{}, third{};
	GetFirstSecondThird(uv, first, second, third); // third will always be 0.000
	float y{ std::stof(second) };
	if (m_IsYAxisInverted) y = 1 - y; // invert Y-axis, bc screen (0, 0) is at the top left but a textures (0, 0) is at the bottom left
	m_UVs.push_back(FVector2{ std::stof(first), y });
}

void ObjParser::GetFirstSecondThird(std::stringstream& fst, std::string& first, std::string& second, std::string& third)
{
	// fst is laid out like this:
	// 1.000 2.000 3.000
	// first being 1.000, second being 2.000 third being 3.000
	std::getline(fst, first, ' ');
	// first now contains 1.000
	std::getline(fst, second, ' ');
	// second now contains 2.000
	std::getline(fst, third, ' ');
	// third now contains 3.000
}

void ObjParser::AssignVertices(std::vector<IVertex>& vertexBuffer, std::vector<unsigned int>& indexBuffer, short& vertexType)
{
	vertexBuffer.clear();
	indexBuffer.clear();
	vertexBuffer.reserve(m_Positions.size());

	const bool isUVs{ (bool)m_UVs.size() };
	const bool isNormals{ (bool)m_Normals.size() };

	vertexType |= isUVs * (int)VertexType::Uv;
	vertexType |= isNormals * (int)VertexType::Norm;

	unsigned int indexCounter{};
	std::vector<Indexed> blackList{};

	if (isUVs && isNormals)
	{
		vertexType |= (int)VertexType::Tan;
		for (size_t i{}; i < m_PositionIndices.size(); ++i) // every possible face
		{
			bool isUnique{ true };
			Indexed index{};
			unsigned int changes{};

			for (Indexed& bl : blackList) // check for blacklisted vertices
			{
				if (bl.v == m_PositionIndices[i]) // v[?] == v[i]
				{
					// same vertex
					if (bl.vt == m_UVIndices[i]) // vt[?] == vt[i]
					{
						// same uv
						if (m_Normals[bl.vn] == m_Normals[m_NormalIndices[i]]) // vn[?] == vn[i]
						{
							// same normal
							index.idx = bl.idx; // save indexCounter
							isUnique = false;
							++changes;
							//continue;
						}
					}
				}
			}

			if (isUnique)
			{
				IVertex v;
				v.p = m_Positions[m_PositionIndices[i]];
				v.uv = m_UVs[m_UVIndices[i]];
				v.n = m_Normals[m_NormalIndices[i]];
				vertexBuffer.push_back(v);
				index.v = m_PositionIndices[i];
				index.vt = m_UVIndices[i];
				index.vn = m_NormalIndices[i];
				index.idx = indexCounter;
				blackList.push_back(index);
				++indexCounter;
			}
			indexBuffer.push_back(index.idx);
		}

		for (uint32_t i{}; i < indexBuffer.size(); i += 3)
		{
			uint32_t idx0{ uint32_t(indexBuffer[i]) };
			uint32_t idx1{ uint32_t(indexBuffer[i + 1]) };
			uint32_t idx2{ uint32_t(indexBuffer[i + 2]) };

			const FPoint3 p0{ vertexBuffer[idx0].p };
			const FPoint3 p1{ vertexBuffer[idx1].p };
			const FPoint3 p2{ vertexBuffer[idx2].p };
			const FVector3 uv0{ vertexBuffer[idx0].uv };
			const FVector3 uv1{ vertexBuffer[idx1].uv };
			const FVector3 uv2{ vertexBuffer[idx2].uv };

			const FVector3 edge0{ p1 - p0 };
			const FVector3 edge1{ p2 - p0 };
			const FVector2 diffX{ FVector2{uv1.x - uv0.x, uv2.x - uv0.x} };
			const FVector2 diffY{ FVector2{uv1.y - uv0.y, uv2.y - uv0.y} };
			float r{ 1.f / Cross(diffX, diffY) };

			FVector3 tangent{ (edge0 * diffY.y - edge1 * diffY.x) * r };
			vertexBuffer[idx0].tan += tangent;
			vertexBuffer[idx1].tan += tangent;
			vertexBuffer[idx2].tan += tangent;
		}
		for (IVertex& vertex : vertexBuffer)
		{
			vertex.tan = GetNormalized(Reject(vertex.tan, vertex.n));
		}
	}
	else if (isUVs && !isNormals)
	{
		for (size_t i{}; i < m_PositionIndices.size(); ++i) // every possible face
		{
			bool isUnique{ true };
			Indexed index{};
			int changes{};

			for (Indexed& bl : blackList) // check for blacklisted vertices
			{
				if (bl.v == m_PositionIndices[i]) // v[?] == v[i]
				{
					// same vertex
					if (bl.vt == m_UVIndices[i]) // vt[?] == vt[i]
					{
						// same uv
						index.idx = bl.idx; // save indexCounter
						isUnique = false;
						++changes;
						//continue;
					}
				}
			}

			if (isUnique)
			{
				IVertex v;
				v.p = m_Positions[m_PositionIndices[i]];
				v.uv = m_UVs[m_UVIndices[i]];
				v.n = m_Normals[m_NormalIndices[i]];
				vertexBuffer.push_back(v);
				index.v = m_PositionIndices[i];
				index.vt = m_UVIndices[i];
				index.vn = m_NormalIndices[i];
				index.idx = indexCounter;
				blackList.push_back(index);
				++indexCounter;
			}
			indexBuffer.push_back(index.idx);
		}
	}
	else if (!isUVs && !isNormals)
	{
		for (size_t i{}; i < m_PositionIndices.size(); ++i) // every possible face
		{
			bool isUnique{ true };
			Indexed index{};

			for (Indexed& bl : blackList) // check for blacklisted vertices
			{
				if (bl.v == m_PositionIndices[i]) // v[?] == v[i]
				{
					// same vertex
					index.idx = bl.idx; // save indexCounter
					isUnique = false;
					break;
				}
			}

			if (isUnique)
			{
				IVertex v;
				v.p = m_Positions[m_PositionIndices[i]];
				v.uv = FVector2{};
				v.c = { 1.f, 1.f, 1.f };
				vertexBuffer.push_back(v);
				index.v = m_PositionIndices[i];
				index.idx = indexCounter;
				blackList.push_back(index);
				++indexCounter;
			}
			indexBuffer.push_back(index.idx);
		}
	}
}

void ObjParser::ClearData()
{
	m_Positions.clear();
	m_UVs.clear();
	m_Normals.clear();
	m_PositionIndices.clear();
	m_UVIndices.clear();
	m_NormalIndices.clear();
}

void ObjParser::SetInvertYAxis(bool value)
{
	m_IsYAxisInverted = value;
}

bool ObjParser::OpenFile(const std::string& filePath)
{
	size_t posOfDot{ filePath.find('.') };
	// no initial value of filePath, means no dot, return false == no opening and no crash either
	if (posOfDot == std::string::npos)
		return false;

	 // not a .obj file!
	const std::string fileExtension = filePath.substr(posOfDot);
	if (filePath.substr(posOfDot) != OBJ_EXTENSION)
	{
		std::cout << "Only \'" << OBJ_EXTENSION << "\' file formats are supported\n Current file extension is : " << fileExtension << '\n';
		return false;
	}

	if (m_ReadFile.is_open())
		CloseFile();

	m_ReadFile.open(filePath);
	return m_ReadFile.is_open();
}

void ObjParser::CloseFile()
{
	m_ReadFile.close();
}