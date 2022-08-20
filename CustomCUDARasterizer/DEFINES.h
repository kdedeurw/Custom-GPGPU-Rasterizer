#pragma once

#ifndef HARDWARE_ACCELERATION
#define HARDWARE_ACCELERATION //enable hardware accelerated CUDA rasterizer
#endif

#ifdef HARDWARE_ACCELERATION

#ifndef BENCHMARK
//#define BENCHMARK //benchmark invidual stages of hardware accelerated cudarenderer
#endif

//show fps in realtime
#ifndef FPS_REALTIME
//#define STATS_REALTIME //can comment this line
#endif

#ifndef BINNING
#define BINNING //enable triangle binning
#ifdef BINNING

#ifndef FINERASTER
//#define FINERASTER //perform fine rasterization per bin
#endif

#endif
#endif

#endif