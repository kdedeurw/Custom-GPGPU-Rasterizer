#include "PCH.h"

#include "CUDASharedMemoryExtended.cuh";

namespace CUDASharedMemoryExtended
{
	const unsigned int TILE_DIM = 32;
	const unsigned int BLOCK_ROWS = 8;
	const unsigned int NUM_REPS = 100;

	__global__ void copy(float* odata, const float* idata)
	{
		const int x = blockIdx.x * TILE_DIM + threadIdx.x;
		const int y = blockIdx.y * TILE_DIM + threadIdx.y;
		const int width = gridDim.x * TILE_DIM;
		//each thread copies 4 elements of the matrix
		//loop iterates through second dimension, so that contiguous threads load and store contiguous data
		//all reads from idata and writes to odata are coalesced
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			odata[(y + j) * width + x] = idata[(y + j) * width + x];
		}
	}

	__global__ void transposeNaive(float* odata, const float* idata)
	{
		const int x = blockIdx.x * TILE_DIM + threadIdx.x;
		const int y = blockIdx.y * TILE_DIM + threadIdx.y;
		const int width = gridDim.x * TILE_DIM;
		//note that only the index for odata is changed
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			odata[x * width + (y + j)] = idata[(y + j) * width + x];
		}
	}

	__global__ void transposeCoalesced(float* odata, const float* idata)
	{
		__shared__ float tile[TILE_DIM][TILE_DIM]; //1024 size

		int x = blockIdx.x * TILE_DIM + threadIdx.x;
		int y = blockIdx.y * TILE_DIM + threadIdx.y;
		const int width = gridDim.x * TILE_DIM;
		//copy contiguous data into rows of shared memory
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
		}

		//sync all threads to counter data races, because of access of different elements operated by different threads
		//(transposed element addresses vs copy, as shown below where x and y are recalculated)
		__syncthreads();

		x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
		y = blockIdx.x * TILE_DIM + threadIdx.y;
		//output contiguous data from rows of shared memory
		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
		}
	}

	__global__ void copySharedMem(float* odata, const float* idata)
	{
		__shared__ float tile[TILE_DIM * TILE_DIM];

		const int x = blockIdx.x * TILE_DIM + threadIdx.x;
		const int y = blockIdx.y * TILE_DIM + threadIdx.y;
		const int width = gridDim.x * TILE_DIM;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];
		}

		//sync all threads to counter data races
		//this is actually not needed, because the operations for an element are performed by the same thread
		//__syncthreads();

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			odata[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
		}
	}

	// No bank-conflict transpose
	// Same as transposeCoalesced except the first tile dimension is padded 
	// to avoid shared memory bank conflicts.
	__global__ void transposeNoBankConflicts(float* odata, const float* idata)
	{
		__shared__ float tile[TILE_DIM][TILE_DIM + 1];

		int x = blockIdx.x * TILE_DIM + threadIdx.x;
		int y = blockIdx.y * TILE_DIM + threadIdx.y;
		const int width = gridDim.x * TILE_DIM;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
		}

		__syncthreads();

		x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
		y = blockIdx.x * TILE_DIM + threadIdx.y;

		for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		{
			odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
		}
	}

	// Check errors and print GB/s
	void postprocess(const float* ref, const float* res, int n, float ms)
	{
		bool passed = true;
		for (int i = 0; i < n; i++)
		{
			if (res[i] != ref[i])
			{
				printf("%d %f %f\n", i, res[i], ref[i]);
				printf("%25s\n", "*** FAILED ***");
				passed = false;
				break;
			}
		}
		if (passed)
		{
			printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
		}
	}

	cudaError_t SharedMemoryExtendedCuda()
	{
		cudaError_t cudaStatus{};

		const int nx = 1024;
		const int ny = 1024;
		const int mem_size = nx * ny * sizeof(float);

		const dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
		const dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

		cudaDeviceProp prop;
		checkCuda(cudaGetDeviceProperties(&prop, 0));
		printf("\nDevice : %s\n", prop.name);
		printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
			nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
		printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
			dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

		checkCuda(cudaSetDevice(0));

		float* h_idata = (float*)malloc(mem_size);
		float* h_cdata = (float*)malloc(mem_size);
		float* h_tdata = (float*)malloc(mem_size);
		float* gold = (float*)malloc(mem_size);

		float* d_idata, * d_cdata, * d_tdata;
		checkCuda(cudaMalloc(&d_idata, mem_size));
		checkCuda(cudaMalloc(&d_cdata, mem_size));
		checkCuda(cudaMalloc(&d_tdata, mem_size));

		// check parameters and calculate execution configuration
		if (nx % TILE_DIM || ny % TILE_DIM)
		{
			printf("nx and ny must be a multiple of TILE_DIM\n");
			goto error_exit;
		}

		if (TILE_DIM % BLOCK_ROWS)
		{
			printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
			goto error_exit;
		}

		// host
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				h_idata[j * nx + i] = j * nx + i;
			}
		}

		// correct result for error checking
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				gold[j * nx + i] = h_idata[i * nx + j];
			}
		}

		// device
		checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

		// events for timing
		cudaEvent_t startEvent, stopEvent;
		checkCuda(cudaEventCreate(&startEvent));
		checkCuda(cudaEventCreate(&stopEvent));
		float ms;

		// ------------
		// time kernels
		// ------------
		printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

		// ----
		// copy 
		// ----
		printf("%25s", "copy");
		checkCuda(cudaMemset(d_cdata, 0, mem_size));
		// warm up
		copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
		checkCuda(cudaEventRecord(startEvent, 0));

		for (int i = 0; i < NUM_REPS; i++)
		{
			copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
		}

		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
		postprocess(h_idata, h_cdata, nx * ny, ms);

		// -------------
		// copySharedMem 
		// -------------
		printf("%25s", "shared memory copy");
		checkCuda(cudaMemset(d_cdata, 0, mem_size));
		// warm up
		copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
		checkCuda(cudaEventRecord(startEvent, 0));

		for (int i = 0; i < NUM_REPS; i++)
		{
			copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
		}

		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
		postprocess(h_idata, h_cdata, nx * ny, ms);

		// --------------
		// transposeNaive 
		// --------------
		printf("%25s", "naive transpose");
		checkCuda(cudaMemset(d_tdata, 0, mem_size));
		// warmup
		transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
		checkCuda(cudaEventRecord(startEvent, 0));

		for (int i = 0; i < NUM_REPS; i++)
		{
			transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
		}

		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
		postprocess(gold, h_tdata, nx * ny, ms);

		// ------------------
		// transposeCoalesced 
		// ------------------
		printf("%25s", "coalesced transpose");
		checkCuda(cudaMemset(d_tdata, 0, mem_size));
		// warmup
		transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
		checkCuda(cudaEventRecord(startEvent, 0));

		for (int i = 0; i < NUM_REPS; i++)
		{
			transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
		}

		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
		postprocess(gold, h_tdata, nx * ny, ms);

		// ------------------------
		// transposeNoBankConflicts
		// ------------------------
		printf("%25s", "conflict-free transpose");
		checkCuda(cudaMemset(d_tdata, 0, mem_size));
		// warmup
		transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
		checkCuda(cudaEventRecord(startEvent, 0));

		for (int i = 0; i < NUM_REPS; i++)
		{
			transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
		}

		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
		postprocess(gold, h_tdata, nx * ny, ms);

	error_exit:
		// cleanup
		checkCuda(cudaEventDestroy(startEvent));
		checkCuda(cudaEventDestroy(stopEvent));
		checkCuda(cudaFree(d_tdata));
		checkCuda(cudaFree(d_cdata));
		checkCuda(cudaFree(d_idata));
		free(h_idata);
		free(h_tdata);
		free(h_cdata);
		free(gold);

		return cudaStatus;
	}
}