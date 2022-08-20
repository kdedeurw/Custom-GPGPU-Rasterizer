#include "PCH.h"
#include "Resolution.h"

const Resolution::UInt2 Resolution::Sizes[] = {
	{ 320U, 240U },
	{ 640U, 480U },
	{ 1280U, 720U },
	{ 1920U, 1080U },
	{ 2560U, 1440U },
	{ 3840U, 2160U } };

const Resolution::UInt2 Resolution::AspectRatios[] = {
	{ 4, 3 },
	{ 4, 3 },
	{ 16, 9 },
	{ 16, 9 },
	{ 16, 9 },
	{ 16, 9 } };