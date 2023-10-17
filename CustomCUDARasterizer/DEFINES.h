#pragma once

//show fps in realtime
#ifndef STATS_REALTIME
#define STATS_REALTIME //can comment this line
#endif

#ifndef STATS_REALTIME
#ifndef BENCHMARK
//#define BENCHMARK //benchmark invidual stages of hardware accelerated cudarenderer
#endif
#endif

#ifndef BINNING
#define BINNING //enable triangle binning
#ifdef BINNING

#ifndef BINMULTIPLIER
#define BINMULTIPLIER 2; //how many bins will be used in regard to the resolution standard (4:3 by default, so (4 * 3 * (1 << BINMULTIPLIER)) bins)
#endif
#ifndef BINQUEUEMAXSIZE
#define BINQUEUEMAXSIZE 256; //how many triangles can fit into a single bin queue
#endif

#ifndef FINERASTER
#define FINERASTER //perform fine rasterization per bin
#endif

#ifdef FINERASTER
#ifndef FINERASTER_SHAREDMEM
//#define FINERASTER_SHAREDMEM
#endif
#endif

#endif
#endif