#pragma once
//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPUHelpers.h"

#include "Mesh.h"

struct WindowHelper;
class Camera;
struct SDL_Surface;

//Kernel functions that executes in parallel
GPU_KERNEL void RenderKernel(SDL_Surface* pBackBuffer, uint32_t width, uint32_t height, float* depthBuffer, Mesh::Textures* pTextures);

//Global functions that launches the kernel
CPU_CALLABLE void Render(WindowHelper* pWindowHelper, Camera* pCamera);