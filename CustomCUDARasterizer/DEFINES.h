#pragma once

//enable hardware accelerated CUDA rasterizer
#ifndef HARDWARE_ACCELERATION
#define HARDWARE_ACCELERATION //can comment this line
#endif

#ifdef HARDWARE_ACCELERATION

//benchmark invidual stages of hardware accelerated cudarenderer
#ifndef BENCHMARK
//#define BENCHMARK //can comment this line
#endif

//show fps in realtime
#ifndef FPS_REALTIME
//#define STATS_REALTIME //can comment this line
#endif

#ifndef BINNING
#define BINNING //can comment this line
#endif

#endif