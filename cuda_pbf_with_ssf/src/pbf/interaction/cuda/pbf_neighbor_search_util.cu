#include "pbf_neighbor_search_util.h"
#include "pbf_grid.h"
#include "../../util/pbf_cuda_util.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <sm_35_intrinsics.h>

#include <cub\cub.cuh>
#include <cub\device\device_radix_sort.cuh>

__constant__ scalar_t c_h;
__constant__ uint32_t c_max_pair_num;
__constant__ scalar_t c_cell_width;
__constant__ uint3 c_grid_size;

namespace {

__global__ void calcHashCUDA(
	uint32_t* hash_particle,
	uint32_t* index_particle,
	const dom_dim* position,
	scalar_t cell_width,
	dom_udim grid_size,
	uint32_t num_particle)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;
	
	auto p = position[index];
	auto grid_pos = pbf::cuda::calcGridPos(p, cell_width);
	auto hash = pbf::cuda::calcGridHash(grid_pos, grid_size);

	hash_particle[index] = hash;
	index_particle[index] = index;
}

__global__ void findCellStartCUDA(
	uint32_t* cell_start, uint32_t* cell_end,
	const uint32_t* hash_particle, const uint32_t* index_particle,
	uint32_t num_particle
	)
{
	extern __shared__ uint32_t shared_hash[];    // blockSize + 1 elements
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t hash;
	// handle case when no. of particles not multiple of block size
	if (index < num_particle) {
		hash = hash_particle[index];

		// Load hash data into shared memory so that we can look at neighboring particle's hash value without loading
		// two hash values per thread
		shared_hash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			shared_hash[0] = hash_particle[index - 1];
		}
	}

	__syncthreads();

	if (index < num_particle) {
		// If this particle has a different cell index to the previous particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of the previous particle's cell

		if (index == 0 || hash != shared_hash[threadIdx.x]) {
			cell_start[hash] = index;

			if (index > 0)
				cell_end[shared_hash[threadIdx.x]] = index;
		}

		if (index == num_particle - 1) {
			cell_end[hash] = index + 1;
		}
	}
}

__global__ void findCellStartCUDA(
	uint32_t* cell_start, uint32_t* cell_end,
	const uint64_t* hash_particle, const uint32_t* index_particle,
	uint32_t num_particle
	)
{
	extern __shared__ uint64_t shared_hash64[];    // blockSize + 1 elements
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	uint64_t hash;
	// handle case when no. of particles not multiple of block size
	if (index < num_particle) {
		hash = hash_particle[index];

		// Load hash data into shared memory so that we can look at neighboring particle's hash value without loading
		// two hash values per thread
		shared_hash64[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			shared_hash64[0] = hash_particle[index - 1];
		}
	}

	__syncthreads();

	if (index < num_particle) {
		// If this particle has a different cell index to the previous particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of the previous particle's cell

		if (index == 0 || hash != shared_hash64[threadIdx.x]) {
			cell_start[hash] = index;

			if (index > 0)
				cell_end[shared_hash64[threadIdx.x]] = index;
		}

		if (index == num_particle - 1) {
			cell_end[hash] = index + 1;
		}
	}
}

__global__ void reorderDataCUDA(
	dom_dim* sorted_pos,
	dom_dim* sorted_vel,
	const dom_dim* old_pos,
	const dom_dim* old_vel,
	const uint32_t* index_particle,
	uint32_t num_particle)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;
	uint32_t index_before_sort = index_particle[index];

	sorted_pos[index] = old_pos[index_before_sort];
	sorted_vel[index] = old_vel[index_before_sort];
}

__global__ void reorderDataCUDA(
	dom_dim* sorted,
	const dom_dim* old,
	const uint32_t* index_particle,
	uint32_t num)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num) return;
	uint32_t index_before_sort = index_particle[index];

	sorted[index] = old[index_before_sort];
}

__global__ void restoreOrderCUDA(
	dom_dim* restored_pos,
	dom_dim* restored_vel,
	const dom_dim* sorted_pos,
	const dom_dim* sorted_vel,
	const uint32_t* index_particle,
	uint32_t num_particle)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;
	uint32_t index_before_sort = index_particle[index];

	restored_pos[index_before_sort] = sorted_pos[index];
	restored_vel[index_before_sort] = sorted_vel[index];
}

}	// end of unnamed ns

namespace pbf {
namespace cuda {

void getRadixSortStorageSize(
	size_t& temp_storage_size,
	uint32_t num_item
	)
{
	uint32_t* key_in = nullptr;
	uint32_t* key_out = nullptr;
	uint32_t* val_in = nullptr;
	uint32_t* val_out = nullptr;
	cub::DeviceRadixSort::SortPairs(NULL, temp_storage_size, key_in, key_out, val_in, val_out, num_item);
}

void sortHashIndexCUB(
	void* temp_storage, size_t temp_storage_size,
	uint32_t* hash, uint32_t* sorted_hash,
	uint32_t* index, uint32_t* sorted_index,
	uint32_t num_particle
	)
{
	cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size,
		hash, sorted_hash, index, sorted_index, num_particle);
}

void calcHash(
	uint32_t* hash, uint32_t* index,
	const dom_dim* position,
	scalar_t cell_width,
	dom_udim grid_size,
	uint32_t num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 256, num_block, num_thread);

	if (num_block > 0)
		calcHashCUDA << < num_block, num_thread >> >
		(hash, index, position, cell_width, grid_size, num_particle);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

// z-order hash
namespace {
__global__ void calcZOrderHashCUDA(
	uint64_t* hash_particle,
	uint32_t* index_particle,
	const dom_dim* position,
	scalar_t cell_width,
	dom_udim grid_size,
	uint32_t num_particle)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto p = position[index];
	auto grid_pos = pbf::cuda::calcGridPos(p, cell_width);
	auto hash = pbf::cuda::calcGridHashZOrder(grid_pos);

	hash_particle[index] = hash;
	index_particle[index] = index;
}
}

void calcZOrderHash(
	uint64_t* hash, uint32_t* index,
	const dom_dim* position,
	scalar_t cell_width,
	dom_udim grid_size,
	uint32_t num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 256, num_block, num_thread);

	if (num_block > 0)
		calcZOrderHashCUDA<< < num_block, num_thread >> >
		(hash, index, position, cell_width, grid_size, num_particle);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void sortHashIndex(
	uint32_t* hash, uint32_t* index,
	uint32_t num_particle)
{
	if (num_particle > 0) {
		thrust::sort_by_key(thrust::device_ptr<uint32_t>(hash), thrust::device_ptr<uint32_t>(hash + num_particle), thrust::device_ptr<uint32_t>(index));
	}
	//thrust::sort_by_key(hash, hash + num_particle, index);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void findCellStart(
	uint32_t* cell_start, uint32_t* cell_end,
	const uint32_t* hash, const uint32_t* index,
	uint32_t num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);
	uint32_t smem_size = sizeof(uint32_t) * (num_thread + 1);

	if (num_block > 0)
		findCellStartCUDA << < num_block, num_thread, smem_size >> >
		(cell_start, cell_end, hash, index, num_particle);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void findCellStart(
	uint32_t* cell_start, uint32_t* cell_end,
	const uint64_t* hash, const uint32_t* index,
	uint32_t num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);
	uint32_t smem_size = sizeof(uint64_t) * (num_thread + 1);

	if (num_block > 0)
		findCellStartCUDA << < num_block, num_thread, smem_size >> >
		(cell_start, cell_end, hash, index, num_particle);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void reorderData(
	pbf_phase_array& sorted,
	const pbf_phase_array& old,
	const uint32_t* index,
	uint32_t num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 192, num_block, num_thread);

	if (num_block > 0)
		reorderDataCUDA << < num_block, num_thread >> >(sorted.x, sorted.v,  old.x, old.v, index, num_particle);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void reorderData(
	dom_dim* sorted,
	const dom_dim* old,
	const uint32_t* index,
	uint32_t num)
{
	uint32_t num_thread, num_block;
	computeGridSize(num, 192, num_block, num_thread);

	if (num_block > 0)
		reorderDataCUDA << < num_block, num_thread >> >(sorted, old, index, num);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void restoreOrder(
	pbf_phase_array& restored,
	const pbf_phase_array& sorted,
	const uint32_t* index,
	uint32_t num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 192, num_block, num_thread);

	if (num_block > 0)
		restoreOrderCUDA << < num_block, num_thread >> >(restored.x, restored.v, sorted.x, sorted.v, index, num_particle);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

namespace {
template<typename T>
__device__ void findPair(T* pair_indices, uint16_t& pair_cnt,
	uint32_t pair_index, const dom_dim& self_pos, const dom_dim*  __restrict__ other_pos)
{
	const auto pair_pos = other_pos[pair_index];
	if (pair_cnt < c_max_pair_num-1) {
#if 1
		const auto h2 = c_h * c_h;
		const auto pos_diff = self_pos - pair_pos;
		const auto r2 = glm::dot(pos_diff, pos_diff);
		if (r2 < h2) {
			pair_indices[pair_cnt * 32] = pair_index;	// global
			//pair_indices[pair_cnt * 128] = pair_index;	// shared
			++pair_cnt;
			//printf("here we come\n");
		}
#else
		// set pair anyway
		pair_indices[pair_cnt * 32] = pair_index;
		++pair_cnt;
#endif
	}
}

template<typename T>
__device__ void findPair(T* pair_indices, uint16_t& pair_cnt,
	uint32_t pair_index, const dom_dim& self_pos, const dom_dim& pair_pos)
{
	if (pair_cnt < c_max_pair_num - 1) {
#if 1
		const auto h2 = c_h * c_h;
		const auto pos_diff = self_pos - pair_pos;
		//const auto r2 = glm::dot(pos_diff, pos_diff);
		const auto r2 = fmaf(pos_diff.z, pos_diff.z, fmaf(pos_diff.y, pos_diff.y, pos_diff.x * pos_diff.x));
		if (r2 < h2) {
			pair_indices[pair_cnt * 32] = pair_index;	// global
			//pair_indices[pair_cnt * 128] = pair_index;	// shared
			++pair_cnt;
			//printf("here we come\n");
		}
#else
		// set pair anyway
		pair_indices[pair_cnt * 32] = pair_index;
		++pair_cnt;
#endif
	}
}

template<typename T>
__device__ void searchCell(
	T* pair_indices, uint16_t& pair_cnt,
	uint32_t start_index, uint32_t end_index,
	const dom_dim& self_pos, const dom_dim*  __restrict__ other_pos)
{
	if (start_index != 0xFFFFFFFF) {
		// iterate over perticles in this cell
		for (auto i = start_index; i < end_index; i += 1) {
			// position loading
			const dom_dim pair_pos_0 = other_pos[i];
			 
			findPair(pair_indices, pair_cnt, i, self_pos, pair_pos_0);
			//findPair(pair_indices, pair_cnt, i, self_pos, other_pos);
		}
	}
}

template<typename T>
__device__ void searchGrid(
	T* pair_indices, uint16_t& pair_cnt,
	const uint32_t* cell_start, const uint32_t* cell_end,
	const dom_dim& self_pos, const dom_dim*  __restrict__ other_pos)
{
	auto grid = pbf::cuda::calcGridPos(self_pos, c_cell_width);

	auto sum_prop = dom_dim(0.f);
#pragma unroll
	for (int z = -1; z <= 1; ++z) {
#pragma unroll
		for (int y = -1; y <= 1; ++y) {
#pragma unroll
			for (int x = -1; x <= 1; ++x) {
				dom_idim neighbor_grid(grid.x + x, grid.y + y, grid.z + z);
				auto neighbor_grid_hash = pbf::cuda::calcGridHash(neighbor_grid, c_grid_size);
				auto start_index = cell_start[neighbor_grid_hash];
				auto end_index = cell_end[neighbor_grid_hash];
				searchCell(pair_indices, pair_cnt, start_index, end_index, self_pos, other_pos);
			}
		}
	}
	//printf("pair_cnt: %d\n", pair_cnt);
}

template<typename T>
__global__ void detectNeighborsCUDA(
	T* neighbor_list,
	const dom_dim*  __restrict__ position,
	const uint32_t* paricle_hash, const uint32_t* particle_index,
	const uint32_t* cell_start, const uint32_t* cell_end,
	uint32_t num_particle
	)
{
	//__shared__ uint16_t pair_indices_shared[25 * 128];	// [max number of pair particle * thread number]
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle)
		return;

	const auto self_pos = position[index];
	uint16_t pair_cnt = 0;
	T* neighbot_list_local = neighbor_list + (index % 32) + (index / 32) * 32 * c_max_pair_num;
	searchGrid(neighbot_list_local, pair_cnt, cell_start, cell_end, self_pos, position);

	//printf("pair_cnt: %d\n", pair_cnt);

	//if (pair_cnt > 25)
	//	pair_cnt = 25;
	//__syncthreads();
	//for (uint32_t i = 0; i < pair_cnt; ++i) {
	//	neighbot_list_local[i * 32] = (pair_indices_shared + threadIdx.x)[i * 128];
	//}
}
}	// end of unnamed ns

namespace {
namespace opt {
__device__ void findPair(uint32_t* pair_indices, uint16_t& pair_cnt,
	uint32_t pair_index, const float4& self_pos, const float4* other_pos)
{
	const float4 pair_pos = other_pos[pair_index];
	if (pair_cnt < c_max_pair_num - 1) {
#if 1
		const auto h2 = c_h * c_h;
		const auto pos_diff = glm::vec3(self_pos.x - pair_pos.x, self_pos.y - pair_pos.y, self_pos.z - pair_pos.z);
		const auto r2 = glm::dot(pos_diff, pos_diff);
		if (r2 < h2) {
			pair_indices[pair_cnt * 32] = pair_index;	// global
			//pair_indices[pair_cnt * 128] = pair_index;	// shared
			++pair_cnt;
			//printf("here we come\n");
		}
#else
		// set pair anyway
		pair_indices[pair_cnt * 32] = pair_index;
		++pair_cnt;
#endif
	}
}

__device__ void searchCell(
	uint32_t* pair_indices, uint16_t& pair_cnt,
	uint32_t start_index, uint32_t end_index,
	const float4& self_pos, const float4* other_pos)
{
	if (start_index != 0xFFFFFFFF) {
		// iterate over perticles in this cell
		for (auto i = start_index; i < end_index; ++i) {
			opt::findPair(pair_indices, pair_cnt, i, self_pos, other_pos);
		}
		//printf("particle in a cell: %d\n", end_index - start_index);
	}
}

__device__ void searchGrid(
	uint32_t* pair_indices, uint16_t& pair_cnt, uint32_t* cell_index_shared,
	const uint32_t* cell_start, const uint32_t* cell_end,
	const float4& self_pos, const float4* other_pos)
{
	auto grid = pbf::cuda::calcGridPos(self_pos, c_cell_width);

	auto sum_prop = dom_dim(0.f);
#pragma unroll
	for (int z = -1; z <= 1; ++z) {
#pragma unroll
		for (int y = -1; y <= 1; ++y) {
#if 1
#pragma unroll
			for (int x = -1; x <= 1; ++x) {
				dom_idim neighbor_grid(grid.x + x, grid.y + y, grid.z + z);
				auto neighbor_grid_hash = pbf::cuda::calcGridHash(neighbor_grid, c_grid_size);
				cell_index_shared[0 + 128 * (x + 1)] = cell_start[neighbor_grid_hash];
				cell_index_shared[0 + 128 * (x + 1) + 128 * 3] = cell_end[neighbor_grid_hash];
			}
#pragma unroll
			for (int x = -1; x <= 1; ++x) {
				auto start_index = cell_index_shared[0 + 128 * (x + 1)];
				auto end_index = cell_index_shared[0 + 128 * (x + 1) + 128 * 3];
				opt::searchCell(pair_indices, pair_cnt, start_index, end_index, self_pos, other_pos);
			}
#else
#pragma unroll
			// no shared
			for (int x = -1; x <= 1; ++x) {
				dom_idim neighbor_grid(grid.x + x, grid.y + y, grid.z + z);
				auto neighbor_grid_hash = pbf::cuda::calcGridHash(neighbor_grid, c_grid_size);
				//auto start_index = __ldg(&cell_start[neighbor_grid_hash]);
				//auto end_index = __ldg(&cell_end[neighbor_grid_hash]);
				auto start_index = cell_start[neighbor_grid_hash];
				auto end_index = cell_end[neighbor_grid_hash];
				opt::searchCell(pair_indices, pair_cnt, start_index, end_index, self_pos, other_pos);
			}
#endif
		}
	}
	//printf("pair_cnt: %d\n", pair_cnt);
}

__global__ void detectNeighborsCUDA(
	uint32_t* neighbor_list,
	const float4* position,
	const uint32_t* paricle_hash, const uint32_t* particle_index,
	const uint32_t* cell_start, const uint32_t* cell_end,
	uint32_t num_particle
	)
{
	__shared__ uint32_t cell_index[3 * 2 * 128];
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle)
		return;

	const auto self_pos = position[index];
	uint16_t pair_cnt = 0;
	uint32_t* neighbot_list_local = neighbor_list + (index % 32) + (index / 32) * 32 * c_max_pair_num;
	opt::searchGrid(neighbot_list_local, pair_cnt, cell_index + threadIdx.x, cell_start, cell_end, self_pos, position);
}
} // end of opt ns
}	// end of unnamed ns

namespace {
namespace zorder {
inline
__device__ void findPair(uint16_t* pair_indices, uint16_t& pair_cnt,
	uint32_t pair_index, const float4& self_pos, const float4* other_pos)
{
	const float4 pair_pos = other_pos[pair_index];
	if (pair_cnt < c_max_pair_num - 1) {
#if 1
		const float h2 = c_h * c_h;
		const auto pos_diff = make_float3(self_pos.x - pair_pos.x, self_pos.y - pair_pos.y, self_pos.z - pair_pos.z);
		const auto r2 = pos_diff.x * pos_diff.x + pos_diff.y * pos_diff.y + pos_diff.z * pos_diff.z;
		if (r2 < h2) {
			//pair_indices[pair_cnt * 32] = pair_index;	// global
			pair_indices[pair_cnt * 128] = pair_index;	// shared
			++pair_cnt;
			//printf("here we come\n");
		}

#else
		// set pair anyway
		pair_indices[pair_cnt * 32] = pair_index;
		++pair_cnt;
#endif
	}
}

inline
__device__ void findPair(uint16_t* pair_indices, uint16_t& pair_cnt,
uint32_t pair_index, const float4& self_pos, const float4& pair_pos)
{
	if (pair_cnt < c_max_pair_num - 1) {
#if 1
		const float h2 = c_h * c_h;
		const auto pos_diff = glm::vec3(self_pos.x - pair_pos.x, self_pos.y - pair_pos.y, self_pos.z - pair_pos.z);
		const auto r2 = glm::dot(pos_diff, pos_diff); 
		//const auto pos_diff = make_float3(self_pos.x - pair_pos.x, self_pos.y - pair_pos.y, self_pos.z - pair_pos.z);
		//const auto r2 = pos_diff.x * pos_diff.x + pos_diff.y * pos_diff.y + pos_diff.z * pos_diff.z;
		if (r2 < h2) {
			pair_indices[pair_cnt * 32] = pair_index;	// global
			//pair_indices[pair_cnt * 128] = pair_index;	// shared
			++pair_cnt;
			//printf("here we come\n");
		}

#else
		// set pair anyway
		pair_indices[pair_cnt * 32] = pair_index;
		++pair_cnt;
#endif
	}
}

inline
__device__ void searchCell(
	uint16_t* pair_indices, uint16_t& pair_cnt,
	const uint32_t* cell_start, const uint32_t* cell_end,
	dom_idim neighbor_grid,
	const float4& self_pos, const float4* other_pos)
{
	auto neighbor_grid_hash = pbf::cuda::calcGridHashZOrder(neighbor_grid);
	//auto start_index = __ldg(&cell_start[neighbor_grid_hash]);
	//auto end_index = __ldg(&cell_end[neighbor_grid_hash]);
	auto start_index = cell_start[neighbor_grid_hash];
	auto end_index = cell_end[neighbor_grid_hash];

	if (start_index != 0xFFFFFFFF) {
		// iterate over perticles in this cell
		for (auto i = start_index; i < end_index; i += 1) {
			// position loading
			const float4 pair_pos_0 = other_pos[i];
			//float4 pair_pos_1;
			//if (i < end_index - 1)
			//	pair_pos_1 = other_pos[i + 1];
			//float4 pair_pos_2;
			//if (i < end_index - 2)
			//	pair_pos_2 = other_pos[i + 2];
			// 
			zorder::findPair(pair_indices, pair_cnt, i, self_pos, pair_pos_0);
			//if (i < end_index - 1)
			//	zorder::findPair(pair_indices, pair_cnt, i + 1, self_pos, pair_pos_1);
			//if (i < end_index - 2)
			//	zorder::findPair(pair_indices, pair_cnt, i + 2, self_pos, pair_pos_2);
		}
		//printf("particle in a cell: %u\n", end_index - start_index);
	}
}

inline
__device__ void searchGrid(
	uint16_t* pair_indices, uint16_t& pair_cnt,
	const uint32_t* cell_start, const uint32_t* cell_end,
	const float4& self_pos, const float4* other_pos)
{
	auto grid = pbf::cuda::calcGridPos(self_pos, c_cell_width);
	//if (grid.x < 1 || grid.y < 1 || grid.z < 1)
		//printf("%d, %d, %d\n", grid.x, grid.y, grid.z);
#if 1
#pragma unroll
	for (int z = -1; z <= 1; ++z) {
#pragma unroll
		for (int y = -1; y <= 1; ++y) {
#pragma unroll
			for (int x = -1; x <= 1; ++x) {
				dom_idim neighbor_grid(grid.x + x, grid.y + y, grid.z + z);
				zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
			}
		}
	}
#else
	{	// 0
		dom_idim neighbor_grid(grid.x - 1, grid.y - 1, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 1
		dom_idim neighbor_grid(grid.x - 0, grid.y - 1, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 2
		dom_idim neighbor_grid(grid.x - 1, grid.y - 0, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 3
		dom_idim neighbor_grid(grid.x - 0, grid.y - 0, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 4
		dom_idim neighbor_grid(grid.x - 1, grid.y - 1, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 5
		dom_idim neighbor_grid(grid.x - 0, grid.y - 1, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 6
		dom_idim neighbor_grid(grid.x - 1, grid.y - 0, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 7
		dom_idim neighbor_grid(grid.x - 0, grid.y - 0, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 8
		dom_idim neighbor_grid(grid.x + 1, grid.y - 1, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 9
		dom_idim neighbor_grid(grid.x + 1, grid.y - 0, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 10
		dom_idim neighbor_grid(grid.x + 1, grid.y - 1, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 11
		dom_idim neighbor_grid(grid.x + 1, grid.y - 0, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 12
		dom_idim neighbor_grid(grid.x - 1, grid.y + 1, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 13
		dom_idim neighbor_grid(grid.x - 0, grid.y + 1, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 14
		dom_idim neighbor_grid(grid.x - 1, grid.y + 1, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 15
		dom_idim neighbor_grid(grid.x - 0, grid.y + 1, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 16
		dom_idim neighbor_grid(grid.x + 1, grid.y + 1, grid.z - 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 17
		dom_idim neighbor_grid(grid.x + 1, grid.y + 1, grid.z - 0);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 18
		dom_idim neighbor_grid(grid.x - 1, grid.y - 1, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 19
		dom_idim neighbor_grid(grid.x - 0, grid.y - 1, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 20
		dom_idim neighbor_grid(grid.x - 1, grid.y - 0, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 21
		dom_idim neighbor_grid(grid.x - 0, grid.y - 0, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 22
		dom_idim neighbor_grid(grid.x + 1, grid.y - 1, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 23
		dom_idim neighbor_grid(grid.x + 1, grid.y - 0, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 24
		dom_idim neighbor_grid(grid.x - 1, grid.y + 1, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 25
		dom_idim neighbor_grid(grid.x - 0, grid.y + 1, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}
	{	// 26
		dom_idim neighbor_grid(grid.x + 1, grid.y + 1, grid.z + 1);
		zorder::searchCell(pair_indices, pair_cnt, cell_start, cell_end, neighbor_grid, self_pos, other_pos);
	}

#endif
	//printf("pair_cnt: %d\n", pair_cnt);
}

__global__ void detectNeighborsCUDA(
	uint16_t* neighbor_list,
	const float4* position,
	const uint32_t* cell_start, const uint32_t* cell_end,
	uint32_t num_particle
	)
{
	__shared__ uint16_t pair_indices_shared[25 * 128];	// [max number of pair particle * thread number]
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle)
		return;

	const auto self_pos = position[index];
	uint16_t pair_cnt = 0;

	// shared
	//uint16_t* pair_indices = &pair_indices_shared[threadIdx.x];
	//zorder::searchGrid(pair_indices, pair_cnt, cell_start, cell_end, self_pos, position);
	//__syncthreads();

	uint16_t* neighbot_list_local = neighbor_list + (index % 32) + (index / 32) * 32 * c_max_pair_num;

	// no shared
	zorder::searchGrid(neighbot_list_local, pair_cnt, cell_start, cell_end, self_pos, position);

	// store from shared
	//for (uint16_t i = 0; i < pair_cnt; ++i) {
	//	neighbot_list_local[i * 32] = (pair_indices_shared)[i * 128 + threadIdx.x];
	//}
	//printf("pair cnt: %u\n", pair_cnt);

}
} // end of zorder ns
} // end of unnamed ns

void detectNeighbors(
	uint32_t* neighbor_list,
	const dom_dim* position,
	const uint32_t* cell_start, const uint32_t* cell_end,
	scalar_t cell_width, dom_udim grid_size,
	scalar_t smoothing_length, uint32_t num_particle,
	uint32_t max_pair_particle_num
	)
{
	// constant memory
	cudaMemcpyToSymbol(c_h, &smoothing_length, sizeof(scalar_t));
	cudaMemcpyToSymbol(c_max_pair_num, &max_pair_particle_num, sizeof(uint32_t));
	cudaMemcpyToSymbol(c_cell_width, &cell_width, sizeof(scalar_t));
	uint3 grid_size_ = make_uint3(grid_size.x, grid_size.y, grid_size.z);
	cudaMemcpyToSymbol(c_grid_size, &grid_size_, sizeof(uint3));

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

#if 1
	if (num_block > 0)
		detectNeighborsCUDA<uint32_t><< < num_block, num_thread >> >
		(neighbor_list, position, nullptr, nullptr, cell_start, cell_end, num_particle);

#else
	// dom_dim to float4
	std::vector<dom_dim> pos_host(num_particle);
	cudaMemcpy(pos_host.data(), position, sizeof(dom_dim) * num_particle, cudaMemcpyDeviceToHost);
	std::vector<float4> pos4_host(num_particle);
	for (uint32_t i = 0; i < num_particle; ++i) {
		pos4_host[i] = make_float4(pos_host[i].x, pos_host[i].y, pos_host[i].z, 0.f);
	}
	float4* pos4;
	cudaMalloc(&pos4, sizeof(float4) * num_particle);
	cudaMemcpy(pos4, pos4_host.data(), sizeof(float4) * num_particle, cudaMemcpyHostToDevice);

	// uint16_t* neighbor_list
	uint16_t* nl16;
	cudaMalloc(&nl16, sizeof(uint16_t) * num_particle * max_pair_particle_num);
	cudaMemset(nl16, 0xFFFF, sizeof(uint16_t) * num_particle * max_pair_particle_num);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// kernel launch
	if (num_block > 0)
		zorder::detectNeighborsCUDA << < num_block, num_thread >> >
		(nl16, pos4, cell_start, cell_end, num_particle);

	// uint16_t to uint32_t
	std::vector<uint32_t> nl_host(num_particle * max_pair_particle_num);
	std::vector<uint16_t> nl16_host(num_particle * max_pair_particle_num);
	cudaMemcpy(nl16_host.data(), nl16, sizeof(uint16_t) * num_particle * max_pair_particle_num, cudaMemcpyDeviceToHost);
	for (uint32_t i = 0; i < num_particle * max_pair_particle_num; ++i) {
		const auto nl_v = nl16_host[i];
		if (nl_v != 0xFFFF)
			nl_host[i] = nl16_host[i];
		else
			nl_host[i] = 0xFFFFFFFF;
	}
	cudaMemcpy(neighbor_list, nl_host.data(), sizeof(uint32_t) * num_particle * max_pair_particle_num, cudaMemcpyHostToDevice);

	cudaFree(nl16);
	cudaFree(pos4);

#endif

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	//exit(0);

}


}	// end of cuda ns
}	// end of pbf ns

