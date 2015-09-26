#pragma once
#include "../pbf_type.h"

namespace pbf {

namespace ns {
	const uint32_t empty = 0xFFFFFFFF;
}

class neighbor_search
{
public:
	neighbor_search(
		uint32_t particle_num,
		scalar_t cell_width,
		dom_udim grid_size
		);

	~neighbor_search();

	// extension of reorderDataAndFindCellStart, neighbor list is constructed
	void detectNeighbors(
		dom_dim* sorted_predicted_position,
		const dom_dim* old_predicted_position,
		dom_dim* sorted_current_position,
		const dom_dim* old_current_position,
		scalar_t smoothing_length,
		uint32_t num_particle);

	void reorderDataAndFindCellStart(
		dom_dim* sorted_predicted_position,
		const dom_dim* old_predicted_position,
		dom_dim* sorted_current_position,
		const dom_dim* old_current_position,
		uint32_t num_particle);

	void restoreOrder(
		pbf_phase_array& restored,
		const pbf_phase_array& sorted,
		uint32_t num_particle);

	void restoreOrder(
		dom_dim* restored,
		const dom_dim* sorted,
		uint32_t num_particle);

	const uint32_t* getHash() { return hash_index.d_hash; }
	const uint32_t* getIndex() { return hash_index.d_index; }
	const uint32_t* getCellStart() { return cell_start; }
	const uint32_t* getCellEnd() { return cell_end; }
	const uint32_t* getNeighborList() { return neighbor_list;  }
	scalar_t getCellWidth() { return cell_width; }
	dom_udim getGridSize() { return grid_size; }
	uint32_t getMaxPairParticleNum() { return max_pair_particle_num;  }
	
private:
	uint32_t* neighbor_list;
	hash_index_t hash_index;
	uint32_t* cell_start;
	uint32_t* cell_end;
	scalar_t cell_width;
	dom_udim grid_size;
	uint32_t max_particle_num;
	uint32_t max_pair_particle_num;
};

} // end of pbf ns

