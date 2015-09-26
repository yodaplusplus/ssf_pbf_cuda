#include "../pbf_grid.h"

namespace pbf {
namespace cuda {
namespace detail {

template<typename func_t, typename arg_t>
__host__ __device__
typename ContributionTraits<func_t>::ConT calcContributionWithinCell(
uint32_t start_index,
uint32_t end_index,
func_t& contribute,
arg_t& arg_list
)
{
	//printf("%d, ", end_index - start_index);
	auto sum_prop = ContributionTraits<func_t>::zero();
	if (start_index != 0xFFFFFFFF) {
		// iterate over perticles in this cell
		for (auto i = start_index; i < end_index; ++i) {
			sum_prop += contribute(i, arg_list);
		}
	}
	return sum_prop;
}

template<typename func_t, typename arg_t>
__host__ __device__
typename ContributionTraits<func_t>::ConT calcContribution(
dom_dim self_pos,
const uint32_t* cell_start,
const uint32_t* cell_end,
scalar_t cell_width,
dom_udim grid_size,
func_t& contribute,
arg_t& arg_list)
{
	auto grid = cuda::calcGridPos(self_pos, cell_width);

	auto sum_prop = ContributionTraits<func_t>::zero();
#pragma unroll
	for (int z = -1; z <= 1; ++z) {
#pragma unroll
		for (int y = -1; y <= 1; ++y) {
#pragma unroll
			for (int x = -1; x <= 1; ++x) {
				dom_idim neighbor_grid(grid.x + x, grid.y + y, grid.z + z);
				auto neighbor_grid_hash = cuda::calcGridHash(neighbor_grid, grid_size);
				auto start_index = cell_start[neighbor_grid_hash];
				auto end_index = cell_end[neighbor_grid_hash];
				sum_prop += calcContributionWithinCell(start_index, end_index, contribute, arg_list);
			}
		}
	}

	return sum_prop;
}

#ifdef PBF_2D
template<typename func_t, typename arg_t>
__host__ __device__ typename ContributionTraits<func_t>::ConT calcContribution(
	dom_dim self_pos,
	const uint32_t* cell_start,
	const uint32_t* cell_end,
	scalar_t cell_width,
	dom_udim grid_size,
	func_t& contribute,
	arg_t& arg_list)
{
	auto grid = cuda::detail::calcGridPos(self_pos, cell_width);

	auto sum_prop = ContributionTraits<func_t>::zero();
#pragma unroll
	for (int y = -1; y <= 1; ++y) {
#pragma unroll
		for (int x = -1; x <= 1; ++x) {
			dom_idim neighbor_grid(grid.x + x, grid.y + y);
			auto neighbor_grid_hash = cuda::detail::calcGridHash(neighbor_grid, grid_size);
			auto start_index = cell_start[neighbor_grid_hash];
			auto end_index = cell_end[neighbor_grid_hash];
			sum_prop += calcContributionWithinCell(start_index, end_index, contribute, arg_list);
		}
	}

	return sum_prop;
}
#endif

} // end of detail ns
} // end of cuda ns
} // end of pbf ns
