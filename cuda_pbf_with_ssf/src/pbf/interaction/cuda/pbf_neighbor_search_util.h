#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda {

void getRadixSortStorageSize(
	size_t& temp_storage_size,
	uint32_t num_item
	);

void sortHashIndexCUB(
	void* temp_storage, size_t temp_storage_size,
	uint32_t* hash, uint32_t* sorted_hash,
	uint32_t* index, uint32_t* sorted_index,
	uint32_t num_particle
	);

void calcHash(
	uint32_t* hash, uint32_t* index,
	const dom_dim* position,
	scalar_t cell_width,
	dom_udim grid_size,
	uint32_t num_particle);

void calcZOrderHash(
	uint64_t* hash, uint32_t* index,
	const dom_dim* position,
	scalar_t cell_width,
	dom_udim grid_size,
	uint32_t num_particle);

void sortHashIndex(
	uint32_t* hash, uint32_t* index,
	uint32_t num_particle);

// cell_start, end must be initialized before calling
void findCellStart(
	uint32_t* cell_start, uint32_t* cell_end,
	const uint32_t* hash, const uint32_t* index,
	uint32_t num_particle);

// cell_start, end must be initialized before calling
void findCellStart(
	uint32_t* cell_start, uint32_t* cell_end,
	const uint64_t* hash, const uint32_t* index,
	uint32_t num_particle);

void reorderData(
	pbf_phase_array& sorted,
	const pbf_phase_array& old,
	const uint32_t* index,
	uint32_t num_particle);

void reorderData(
	dom_dim* sorted,
	const dom_dim* old,
	const uint32_t* index,
	uint32_t num);

void restoreOrder(
	pbf_phase_array& restored,
	const pbf_phase_array& sorted,
	const uint32_t* index,
	uint32_t num_particle);

void detectNeighbors(
	uint32_t* neighbor_list,
	const dom_dim* position,
	const uint32_t* cell_start, const uint32_t* cell_end,
	scalar_t cell_width, dom_udim grid_size,
	scalar_t h, uint32_t num_particle,
	uint32_t max_pair_particle_num
	);

}	// end of cuda ns
}	// end of pbf ns
