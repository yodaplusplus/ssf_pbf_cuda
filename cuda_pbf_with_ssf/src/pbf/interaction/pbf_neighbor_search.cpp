#include "pbf_neighbor_search.h"
#include "cuda/pbf_neighbor_search_util.h"
#include "../util/cuda/pbf_fill.h"
#include "../util/pbf_cuda_util.h"

namespace pbf {

neighbor_search::neighbor_search(
	uint32_t arg_num,
	scalar_t arg_cell_width,
	dom_udim arg_grid_size) :
	max_pair_particle_num(120)
{
	max_particle_num = arg_num;
	cell_width = arg_cell_width;
	grid_size = arg_grid_size;
	uint32_t cell_num = grid_size.x * grid_size.y * grid_size.z;
	cudaMalloc(&cell_start, cell_num * sizeof(uint32_t));
	cudaMalloc(&cell_end, cell_num * sizeof(uint32_t));
	cudaMalloc(&hash_index.d_hash, max_particle_num * sizeof(uint32_t));
	cudaMalloc(&hash_index.d_index, max_particle_num * sizeof(uint32_t));
	cudaMalloc(&neighbor_list, max_particle_num * max_pair_particle_num * sizeof(uint32_t));

	cudaMemset(cell_start, 0, cell_num * sizeof(uint32_t));
	cudaMemset(cell_end, 0, cell_num * sizeof(uint32_t));
	cudaMemset(hash_index.d_hash, 0, max_particle_num * sizeof(uint32_t));
	cudaMemset(hash_index.d_index, 0, max_particle_num * sizeof(uint32_t));
	const uint32_t empty = 0xFFFFFFFF;
	pbf::cuda::fill(cell_start, empty, cell_num);
	pbf::cuda::fill(neighbor_list, empty, max_particle_num * max_pair_particle_num);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

}

neighbor_search::~neighbor_search()
{
	cudaFree(cell_start);
	cudaFree(cell_end);
	cudaFree(hash_index.d_hash);
	cudaFree(hash_index.d_index);
	cudaFree(neighbor_list);
}

void neighbor_search::detectNeighbors(
	dom_dim* sorted_predicted_position,
	const dom_dim* old_predicted_position,
	dom_dim* sorted_current_position,
	const dom_dim* old_current_position,
	scalar_t smoothing_length,
	uint32_t num_particle)
{
	const uint32_t empty = 0xFFFFFFFF;
	uint32_t total_cell = grid_size.x * grid_size.y * grid_size.z;
#ifdef PBF_2D
	uint32_t total_cell = grid_size.x * grid_size.y;
#endif
	pbf::cuda::fill(cell_start, empty, total_cell);
	pbf::cuda::fill(neighbor_list, empty, max_pair_particle_num * num_particle);
	if (num_particle > 0){
		pbf::cuda::calcHash(hash_index.d_hash, hash_index.d_index, old_predicted_position, cell_width, grid_size, num_particle);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		pbf::cuda::sortHashIndex(hash_index.d_hash, hash_index.d_index, num_particle);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		pbf::cuda::findCellStart(cell_start, cell_end, hash_index.d_hash, hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_predicted_position, old_predicted_position, hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_current_position, old_current_position, hash_index.d_index, num_particle);
		pbf::cuda::detectNeighbors(neighbor_list, sorted_predicted_position,
			hash_index.d_hash, hash_index.d_index, cell_start, cell_end,
			cell_width, grid_size, smoothing_length, num_particle, max_pair_particle_num);
	}
}


void neighbor_search::reorderDataAndFindCellStart(
	dom_dim* sorted_predicted_position,
	const dom_dim* old_predicted_position,
	dom_dim* sorted_current_position,
	const dom_dim* old_current_position,
	uint32_t num_particle)
{
	const uint32_t empty = 0xFFFFFFFF;
	uint32_t total_cell = grid_size.x * grid_size.y * grid_size.z;
#ifdef PBF_2D
	uint32_t total_cell = grid_size.x * grid_size.y;
#endif
	pbf::cuda::fill(cell_start, empty, total_cell);
	if (num_particle > 0){
		pbf::cuda::calcHash(hash_index.d_hash, hash_index.d_index, old_predicted_position, cell_width, grid_size, num_particle);
		pbf::cuda::sortHashIndex(hash_index.d_hash, hash_index.d_index, num_particle);
		pbf::cuda::findCellStart(cell_start, cell_end, hash_index.d_hash, hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_predicted_position, old_predicted_position, hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_current_position, old_current_position, hash_index.d_index, num_particle);
	}
}

void neighbor_search::restoreOrder(
	pbf_phase_array& restored,
	const pbf_phase_array& sorted,
	uint32_t num_particle)
{
	pbf::cuda::restoreOrder(restored, sorted, hash_index.d_index, num_particle);
}

void neighbor_search::restoreOrder(
	dom_dim* restored,
	const dom_dim* sorted,
	uint32_t num_particle)
{
	std::cerr << "not implemented" << std::endl;
	exit(-1);
}

} // end of pbf ns
