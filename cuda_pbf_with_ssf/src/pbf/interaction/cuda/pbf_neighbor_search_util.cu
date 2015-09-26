#include "pbf_neighbor_search_util.h"
#include "pbf_grid.h"
#include "../../util/pbf_cuda_util.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

void sortHashIndex(
	uint32_t* hash, uint32_t* index,
	uint32_t num_particle)
{
	if (num_particle > 0)
		thrust::sort_by_key(thrust::device_ptr<uint32_t>(hash), thrust::device_ptr<uint32_t>(hash + num_particle), thrust::device_ptr<uint32_t>(index));
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
__device__ void findPair(uint32_t* pair_indices, uint32_t& pair_cnt,
	uint32_t pair_index, const dom_dim& self_pos, const dom_dim* other_pos)
{
	if (pair_cnt < c_max_pair_num-1) {
		const auto pair_pos = other_pos[pair_index];
		const auto h2 = c_h * c_h;
		const auto pos_diff = self_pos - pair_pos;
		const auto r2 = glm::dot(pos_diff, pos_diff);
		if (r2 < h2) {
			pair_indices[pair_cnt * 32] = pair_index;
			++pair_cnt;
			//printf("here we come\n");
		}
	}
}

__device__ void searchCell(
	uint32_t* pair_indices, uint32_t& pair_cnt,
	uint32_t start_index, uint32_t end_index,
	const dom_dim& self_pos, const dom_dim* other_pos)
{
	if (start_index != 0xFFFFFFFF) {
		// iterate over perticles in this cell
		for (auto i = start_index; i < end_index; ++i) {
			findPair(pair_indices, pair_cnt, i, self_pos, other_pos);
		}
	}
}

__device__ void searchGrid(
	uint32_t* pair_indices,
	const uint32_t* cell_start, const uint32_t* cell_end,
	const dom_dim& self_pos, const dom_dim* other_pos)
{
	auto grid = pbf::cuda::calcGridPos(self_pos, c_cell_width);

	uint32_t pair_cnt = 0;
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

__global__ void detectNeighborsCUDA(
	uint32_t* neighbor_list,
	const dom_dim* position,
	const uint32_t* paricle_hash, const uint32_t* particle_index,
	const uint32_t* cell_start, const uint32_t* cell_end,
	uint32_t num_particle
	)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle)
		return;

	const auto self_pos = position[index];
	uint32_t* neighbot_list_local = neighbor_list + (index % 32) + (index / 32) * 32 * c_max_pair_num;
	searchGrid(neighbot_list_local, cell_start, cell_end, self_pos, position);
}
}	// end of unnamed ns


void detectNeighbors(
	uint32_t* neighbor_list,
	const dom_dim* position,
	const uint32_t* hash, const uint32_t* index,
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

	if (num_block > 0)
		detectNeighborsCUDA<<< num_block, num_thread >>>
		(neighbor_list, position, hash, index, cell_start, cell_end, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

}	// end of cuda ns
}	// end of pbf ns

