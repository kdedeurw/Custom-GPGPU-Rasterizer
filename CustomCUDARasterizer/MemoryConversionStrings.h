#pragma once
#include <string>
#include "GPUHelpers.h"

CPU_CALLABLE static
std::string ToKbs(size_t bytes)
{
	const size_t toKbs = 1024;
	std::string output{ std::to_string(bytes / toKbs) + "Kb" };
	return output;
}

CPU_CALLABLE static
std::string ToMbs(size_t bytes)
{
	const size_t toMBs = 1024 * 1024;
	std::string output{ std::to_string(bytes / toMBs) + "Mb" };
	return output;
}

CPU_CALLABLE static
std::string ToGbs(size_t bytes)
{
	const size_t toGBs = 1024 * 1024 * 1024;
	std::string output{ std::to_string(bytes / toGBs) + "Gb" };
	return output;
}