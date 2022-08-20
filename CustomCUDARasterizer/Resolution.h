#pragma once

struct Resolution
{
	struct UInt2
	{
		unsigned int w, h;
	};
	enum ResolutionStandard
	{
		VGA = 0, //320x240
		SD = 1, //640x480
		HD = 2, //1280x720
		FHD = 3, //1920x1080
		QHD = 4, //2560x1440
		UHD = 5, //3840x2160
	};
	unsigned int Width;
	unsigned int Height;
	UInt2 AspectRatio;
	ResolutionStandard Standard;

	static Resolution GetResolution(ResolutionStandard standard)
	{
		Resolution res;
		res.Width = Sizes[standard].w;
		res.Height = Sizes[standard].h;
		res.AspectRatio = AspectRatios[standard];
		res.Standard = standard;
		return res;
	}

const static UInt2 Sizes[];
const static UInt2 AspectRatios[];
};