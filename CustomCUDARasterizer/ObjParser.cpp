#include "PCH.h"
#include "ObjParser.h"
#include <iostream>
#include <sstream>

ObjParser::ObjParser() : ObjParser{ "" } {}

ObjParser::ObjParser(const std::string& filePath)
	: m_ReadFile{}
	, m_pVertices{}
	, m_Positions{}
	, m_PositionIndices{}
	, m_IndexBuffer{}
	, m_UVIndices{}
	, m_NormalIndices{}
{
	OpenFile(filePath);
}

ObjParser::~ObjParser()
{
	if (m_pVertices)
	{
		m_pVertices->clear();
		delete m_pVertices;
	}
	m_pVertices = nullptr;
	CloseFile();
}

void ObjParser::ReadFromObjFile()
{
	if (m_ReadFile.is_open())
	{
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

		std::cout << "\n!Done parsing, creating vertices based on parsed info now!\n";

		AssignVertices(); // create vertices, filled with positions, normals, UV coords (and colours?) all at once (store them all in vertex)

		std::cout << "\n!All done!\n";
	}
	else std::cout << "\n!Unable to open file!\n";
}

void ObjParser::ReadFromObjFile(const std::string& filePath)
{
	if (m_ReadFile.is_open())
	{
		CloseFile();
		OpenFile(filePath);
	}
	else /*if (!m_ReadFile.is_open())*/ OpenFile(filePath);
	
	ReadFromObjFile();
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

	std::vector<int> faceIndexes{};
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

void ObjParser::AssignVertices()
{
	if (m_pVertices)
	{
		m_pVertices->clear();
		delete m_pVertices;
	}
	m_pVertices = new std::vector<IVertex>{};
	m_pVertices->reserve(m_Positions.size());

	bool isUVs{ static_cast<bool>(m_UVs.size()) };
	bool isNormals{ static_cast<bool>(m_Normals.size()) };

	int indexCounter{};
	std::vector<Indexed> blackList{};

	if (isUVs && isNormals)
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
				m_pVertices->push_back(IVertex{ m_Positions[m_PositionIndices[i]], m_UVs[m_UVIndices[i]], m_Normals[m_NormalIndices[i]] });
				index.v = m_PositionIndices[i]; index.vt = m_UVIndices[i]; index.vn = m_NormalIndices[i]; index.idx = indexCounter;
				blackList.push_back(index);
				++indexCounter;
			}
			m_IndexBuffer.push_back(index.idx);
		}

		std::vector<IVertex>& vertices = (*m_pVertices);
		for (uint32_t i{}; i < m_IndexBuffer.size(); i += 3)
		{
			uint32_t idx0{ uint32_t(m_IndexBuffer[i]) };
			uint32_t idx1{ uint32_t(m_IndexBuffer[i + 1]) };
			uint32_t idx2{ uint32_t(m_IndexBuffer[i + 2]) };

			const FPoint3 p0{ vertices[idx0].v };
			const FPoint3 p1{ vertices[idx1].v };
			const FPoint3 p2{ vertices[idx2].v };
			const FVector3 uv0{ vertices[idx0].uv };
			const FVector3 uv1{ vertices[idx1].uv };
			const FVector3 uv2{ vertices[idx2].uv };

			const FVector3 edge0{ p1 - p0 };
			const FVector3 edge1{ p2 - p0 };
			const FVector2 diffX{ FVector2{uv1.x - uv0.x, uv2.x - uv0.x} };
			const FVector2 diffY{ FVector2{uv1.y - uv0.y, uv2.y - uv0.y} };
			float r{ 1.f / Cross(diffX, diffY) };

			FVector3 tangent{ (edge0 * diffY.y - edge1 * diffY.x) * r };
			vertices[idx0].tan += tangent;
			vertices[idx1].tan += tangent;
			vertices[idx2].tan += tangent;
		}
		for (IVertex& vertex : vertices)
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
				m_pVertices->push_back(IVertex{ m_Positions[m_PositionIndices[i]], m_UVs[m_UVIndices[i]], m_Normals[m_NormalIndices[i]] });
				index.v = m_PositionIndices[i]; index.vt = m_UVIndices[i]; index.vn = m_NormalIndices[i]; index.idx = indexCounter;
				blackList.push_back(index);
				++indexCounter;
			}
			m_IndexBuffer.push_back(index.idx);
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
				m_pVertices->push_back(IVertex{ m_Positions[m_PositionIndices[i]], FVector2{} });
				index.v = m_PositionIndices[i]; index.idx = indexCounter;
				blackList.push_back(index);
				++indexCounter;
			}
			m_IndexBuffer.push_back(index.idx);
		}
	}
}

std::vector<IVertex> const* ObjParser::GetVertexBuffer() const
{
	if (!m_pVertices) return nullptr;
	return m_pVertices;
}

const std::vector<int> ObjParser::GetIndexBuffer() const
{
	return m_IndexBuffer;
}

void ObjParser::SetInvertYAxis(bool value)
{
	m_IsYAxisInverted = value;
}

bool ObjParser::OpenFile(const std::string& filePath)
{
	size_t posOfDot{ filePath.find('.') };
	if (posOfDot == std::string::npos) return false; // no initial value of filePath, means no dot, return false == no opening and no crash either
	if (filePath.substr(posOfDot) != ".obj") return false; // not a .obj file!

	if (m_ReadFile.is_open()) return false;

	m_ReadFile.open(filePath);
	return m_ReadFile.is_open();
}

void ObjParser::CloseFile()
{
	m_ReadFile.close();
}