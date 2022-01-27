#pragma once

//enable hardware accelerated CUDA rasterizer
#ifndef HARDWARE_ACCELERATION
#define HARDWARE_ACCELERATION //can comment this line
#endif

//benchmark invidual stages of hardware accelerated cudarenderer
#ifndef BENCHMARK
#ifdef HARDWARE_ACCELERATION
//#define BENCHMARK //can comment this line
#endif
#endif

//show fps in realtime
#ifndef FPS_REALTIME
#ifdef HARDWARE_ACCELERATION
//#define FPS_REALTIME //can comment this line
#endif
#endif