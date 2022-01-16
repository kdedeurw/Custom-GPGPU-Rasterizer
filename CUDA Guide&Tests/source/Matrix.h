#pragma once

template <typename T>
struct Matrix
{
	inline Matrix(const int stride)
		: Stride{ stride }
	{
		Data = new T[stride * stride]{};
	}
	virtual inline ~Matrix()
	{
		delete[] Data;
		//Data = nullptr;
	}

	const int Stride;
	T Data[];
};