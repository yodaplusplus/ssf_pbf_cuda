#include "pbf_neighbor_search.h"
#include "cuda/pbf_neighbor_search_util.h"
#include "../util/cuda/pbf_fill.h"
#include "../util/pbf_cuda_util.h"
#include "../util/cuda/pbf_counting_sort.h"

namespace pbf {

neighbor_search::neighbor_search(
	uint32_t arg_num,
	scalar_t arg_cell_width,
	dom_udim arg_grid_size) :
	max_pair_particle_num(50)
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

	// for cub radix sort
	cudaMalloc(&sorted_hash_index.d_hash, max_particle_num * sizeof(uint32_t));
	cudaMalloc(&sorted_hash_index.d_index, max_particle_num * sizeof(uint32_t));
	//temp_cub_storage_size = 0;
	//cuda::getRadixSortStorageSize(temp_cub_storage_size, max_particle_num);
	//cudaMalloc(&temp_cub_storage, temp_cub_storage_size);

	// for counting sort
	temp_cub_storage_size = 0;
	cuda::getPrefixSumStorageSize(temp_cub_storage_size, cell_num);
	cudaMalloc(&temp_cub_storage, temp_cub_storage_size);

	std::cout << "temp size: " << temp_cub_storage_size << std::endl;

	cudaMalloc(&hash_count, max_particle_num * sizeof(uint32_t));
	if (cudaMalloc(&cell_count, cell_num * sizeof(uint32_t)) != cudaSuccess) {
		std::cerr << "cuda malloc failed" << std::endl;
	}
	if (cudaMalloc(&cumulative_cell_count, cell_num * sizeof(uint32_t)) != cudaSuccess) {
		std::cerr << "cuda malloc failed" << std::endl;
	}

	std::cout << "grid size: " << grid_size.x << ", " << grid_size.y << ", " << grid_size.z << std::endl;

	// z-order hashing
	cudaMalloc(&zorder_hash, max_particle_num * sizeof(uint64_t));
	cudaMalloc(&sorted_zorder_hash, max_particle_num * sizeof(uint64_t));

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

	cudaFree(sorted_hash_index.d_hash);
	cudaFree(sorted_hash_index.d_index);
	cudaFree(temp_cub_storage);

	cudaFree(hash_count);
	cudaFree(cell_count);
	cudaFree(cumulative_cell_count);

	cudaFree(zorder_hash);
	cudaFree(sorted_zorder_hash);
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
#ifdef USE_THRUST
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		pbf::cuda::sortHashIndex(hash_index.d_hash, hash_index.d_index, num_particle);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		pbf::cuda::findCellStart(cell_start, cell_end, hash_index.d_hash, hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_predicted_position, old_predicted_position, hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_current_position, old_current_position, hash_index.d_index, num_particle);
		pbf::cuda::detectNeighbors(neighbor_list, sorted_predicted_position,
			hash_index.d_hash, hash_index.d_index, cell_start, cell_end,
			cell_width, grid_size, smoothing_length, num_particle, max_pair_particle_num);
#else
#ifdef USE_CUB
		pbf::cuda::sortHashIndexCUB(temp_cub_storage, temp_cub_storage_size,
			hash_index.d_hash, sorted_hash_index.d_hash, hash_index.d_index, sorted_hash_index.d_index, num_particle);
		pbf::cuda::findCellStart(cell_start, cell_end, sorted_hash_index.d_hash, sorted_hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_predicted_position, old_predicted_position, sorted_hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_current_position, old_current_position, sorted_hash_index.d_index, num_particle);
		pbf::cuda::detectNeighbors(neighbor_list, sorted_predicted_position,
			sorted_hash_index.d_hash, sorted_hash_index.d_index, cell_start, cell_end,
			cell_width, grid_size, smoothing_length, num_particle, max_pair_particle_num);
#else
		// z-order hashing
		//pbf::cuda::calcZOrderHash(zorder_hash, hash_index.d_index, old_predicted_position, cell_width, grid_size, num_particle);
		//pbf::cuda::fill(hash_count, 0, num_particle);
		//pbf::cuda::fill(cell_count, 0, total_cell);
		//pbf::cuda::countingSort(sorted_zorder_hash, sorted_hash_index.d_index, cumulative_cell_count, cell_count,
		//	hash_count, temp_cub_storage, temp_cub_storage_size, zorder_hash, num_particle, total_cell);
		//pbf::cuda::findCellStart(cell_start, cell_end, sorted_zorder_hash, sorted_hash_index.d_index, num_particle);
		//pbf::cuda::reorderData(sorted_predicted_position, old_predicted_position, sorted_hash_index.d_index, num_particle);
		//pbf::cuda::reorderData(sorted_current_position, old_current_position, sorted_hash_index.d_index, num_particle);
		//pbf::cuda::detectNeighbors(neighbor_list, sorted_predicted_position, cell_start, cell_end,
		//	cell_width, grid_size, smoothing_length, num_particle, max_pair_particle_num);

		// trivial indexing
		pbf::cuda::fill(hash_count, 0, num_particle);
		pbf::cuda::fill(cell_count, 0, total_cell);
		pbf::cuda::countingSort(sorted_hash_index.d_hash, sorted_hash_index.d_index, cumulative_cell_count, cell_count,
			hash_count, temp_cub_storage, temp_cub_storage_size, hash_index.d_hash, num_particle, total_cell);
		pbf::cuda::findCellStart(cell_start, cell_end, sorted_hash_index.d_hash, sorted_hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_predicted_position, old_predicted_position, sorted_hash_index.d_index, num_particle);
		pbf::cuda::reorderData(sorted_current_position, old_current_position, sorted_hash_index.d_index, num_particle);
		pbf::cuda::detectNeighbors(neighbor_list, sorted_predicted_position, cell_start, cell_end,
			cell_width, grid_size, smoothing_length, num_particle, max_pair_particle_num);

#endif
#endif
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
