#include "pbf_counting_sort.h"
#include "../pbf_cuda_util.h"
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <sm_35_intrinsics.h>
#include <thrust\scan.h>
#include <thrust\device_ptr.h>
#include <cub/cub.cuh>

#include <stdio.h>
#include <math.h>
#include <algorithm>

namespace pbf {
namespace cuda {

#pragma region hashing_32bit

// insert and count
namespace {
__global__ void insertAndCountCUDA(
	uint32_t* cell_count, uint32_t* hash_count,
	const uint32_t* hashes,
	uint32_t num_item)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_item) return;

	const auto hash = __ldg(&hashes[index]);
	const auto count = atomicAdd(&cell_count[hash], 1);
	hash_count[index] = count;
}
}

void insertAndCount(
	uint32_t* cell_count, uint32_t* hash_count,
	const uint32_t* hash,
	uint32_t num_item)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_item, 256, num_block, num_thread);

	if (num_block > 0)
		insertAndCountCUDA << < num_block, num_thread >> >
		(cell_count, hash_count, hash, num_item);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void getPrefixSumStorageSize(
	size_t& temp_storage_size,
	uint32_t num_item
	)
{
	uint32_t* val_in = nullptr;
	uint32_t* val_out = nullptr;
	cub::DeviceScan::ExclusiveSum(NULL, temp_storage_size, val_in, val_out, num_item);
}

void prefixSum(
	void* temp_storage,
	size_t temp_storage_size,
	uint32_t* cumulative_cell_count,
	const uint32_t* cell_count,
	uint32_t num_cell
	)
{
	cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_size, cell_count, cumulative_cell_count, num_cell);
}

void prefixSum(
	uint32_t* cell_count,
	uint32_t num_cell
	)
{
	thrust::exclusive_scan(thrust::device_ptr<uint32_t>(cell_count), thrust::device_ptr<uint32_t>(cell_count + num_cell),
		thrust::device_ptr<uint32_t>(cell_count));
}

// counting sort
namespace {
__global__ void sortByCountingCUDA(
	uint32_t* sorted_hash,
	const uint32_t* hashes, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_item) return;

	const auto hash = __ldg(&hashes[index]);
	const auto count = __ldg(&hash_count[index]);
	const auto cell_offset = __ldg(&cumulative_cell_count[hash]);
	const auto addr = cell_offset + count;
	sorted_hash[addr] = hash;
}
}

void sortByCounting(
	uint32_t* sorted_hash,
	const uint32_t* hash, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item
	)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_item, 256, num_block, num_thread);

	if (num_block > 0)
		sortByCountingCUDA << < num_block, num_thread >> >
		(sorted_hash, hash, cumulative_cell_count, hash_count, num_item);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

// counting sort
namespace {
__global__ void sortByCountingCUDA(
	uint32_t* sorted_hash, uint32_t* sorted_index,
	const uint32_t* hashes, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_item) return;

	const auto hash = __ldg(&hashes[index]);
	const auto count = __ldg(&hash_count[index]);
	const auto cell_offset = __ldg(&cumulative_cell_count[hash]);
	const auto addr = cell_offset + count;
	sorted_hash[addr] = hash;
	sorted_index[addr] = index;
}
}

void sortByCounting(
	uint32_t* sorted_hash, uint32_t* sorted_index,
	const uint32_t* hash, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item
	)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_item, 256, num_block, num_thread);

	if (num_block > 0)
		sortByCountingCUDA << < num_block, num_thread >> >
		(sorted_hash, sorted_index, hash, cumulative_cell_count, hash_count, num_item);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void countingSort(
	uint32_t* sorted_hash, uint32_t* sorted_index,
	uint32_t* cell_count, uint32_t* hash_count,
	const uint32_t* hash,
	uint32_t num_item, uint32_t num_cell
	)
{
	insertAndCount(cell_count, hash_count, hash, num_item);
	prefixSum(cell_count, num_cell);
	sortByCounting(sorted_hash, sorted_index, hash, cell_count, hash_count, num_item);
}

void countingSort(
	uint32_t* sorted_hash, uint32_t* sorted_index,
	uint32_t* cumulative_cell_count, uint32_t* cell_count, uint32_t* hash_count,
	void* temp_storage, size_t temp_storage_size,	// storage for scan
	const uint32_t* hash,
	uint32_t num_item, uint32_t num_cell
	)
{
	insertAndCount(cell_count, hash_count, hash, num_item);
	prefixSum(temp_storage, temp_storage_size, cumulative_cell_count, cell_count, num_cell);
	sortByCounting(sorted_hash, sorted_index, hash, cumulative_cell_count, hash_count, num_item);
}

#pragma endregion

// for z-order hash
#pragma region hashing_64bit

namespace {
__global__ void insertAndCountCUDA(
	uint32_t* cell_count, uint32_t* hash_count,
	const uint64_t* hashes,
	uint32_t num_item)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_item) return;

	const auto hash = __ldg(&hashes[index]);
	const auto count = atomicAdd(&cell_count[hash], 1);
	hash_count[index] = count;
}
}

void insertAndCount(
	uint32_t* cell_count, uint32_t* hash_count,
	const uint64_t* hash,
	uint32_t num_item)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_item, 256, num_block, num_thread);

	if (num_block > 0)
		insertAndCountCUDA << < num_block, num_thread >> >
		(cell_count, hash_count, hash, num_item);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

// counting sort
namespace {
__global__ void sortByCountingCUDA(
	uint64_t* sorted_hash, uint32_t* sorted_index,
	const uint64_t* hashes, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_item) return;

	const auto hash = __ldg(&hashes[index]);
	const auto count = __ldg(&hash_count[index]);
	const auto cell_offset = cumulative_cell_count[hash];
	const auto addr = cell_offset + count;
	sorted_hash[addr] = hash;
	sorted_index[addr] = index;
}
}

void sortByCounting(
	uint64_t* sorted_hash, uint32_t* sorted_index,
	const uint64_t* hash, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item
	)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_item, 256, num_block, num_thread);

	if (num_block > 0)
		sortByCountingCUDA << < num_block, num_thread >> >
		(sorted_hash, sorted_index, hash, cumulative_cell_count, hash_count, num_item);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void countingSort(
	uint64_t* sorted_hash, uint32_t* sorted_index,
	uint32_t* cumulative_cell_count, uint32_t* cell_count, uint32_t* hash_count,
	void* temp_storage, size_t temp_storage_size,	// storage for scan
	const uint64_t* hash,
	uint32_t num_item, uint32_t num_cell
	)
{
	insertAndCount(cell_count, hash_count, hash, num_item);
	prefixSum(temp_storage, temp_storage_size, cumulative_cell_count, cell_count, num_cell);
	sortByCounting(sorted_hash, sorted_index, hash, cumulative_cell_count, hash_count, num_item);
}

#pragma endregion

}	// end of cuda ns
}	// end of pbf ns
